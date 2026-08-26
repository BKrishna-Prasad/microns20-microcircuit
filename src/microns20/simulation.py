"""Utilities for provisional BioNet simulation experiments.

Simulation assumptions are intentionally kept separate from the CAVE-derived
structural circuit. The Allen reference run is a software/model control. The
MICrONS single-cell run transfers the same Allen perisomatic parameters onto
one processed MICrONS morphology without changing that morphology.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import h5py
import pandas as pd
import yaml


REQUIRED_ALLEN_FILES = (
    "fit_parameters.json",
    "model_metadata.json",
    "reconstruction.swc",
)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}.")

    return data


def _resolve_project_path(
    project_root: Path,
    value: str | Path,
) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def validate_allen_reference_assets(
    project_root: str | Path,
    config_path: str | Path = "configs/simulations/allen_reference.yaml",
) -> dict[str, Any]:
    """Validate the downloaded Allen reference model."""

    root = Path(project_root).resolve()
    settings = _load_yaml(_resolve_project_path(root, config_path))
    model_cfg = settings["allen_reference"]

    source_dir = _resolve_project_path(
        root,
        model_cfg["source_dir"],
    ).resolve()

    if not source_dir.is_dir():
        raise FileNotFoundError(
            f"Allen reference directory does not exist: {source_dir}"
        )

    missing = [
        name
        for name in REQUIRED_ALLEN_FILES
        if not (source_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Allen reference model is missing required files: {missing}"
        )

    modfiles = sorted((source_dir / "modfiles").glob("*.mod"))
    if not modfiles:
        raise FileNotFoundError(
            f"No NEURON .mod files found under {source_dir / 'modfiles'}."
        )

    metadata = json.loads(
        (source_dir / "model_metadata.json").read_text(encoding="utf-8")
    )
    fit = json.loads(
        (source_dir / "fit_parameters.json").read_text(encoding="utf-8")
    )

    expected_model_id = int(model_cfg["model_id"])
    observed_model_id = int(metadata["id"])

    if observed_model_id != expected_model_id:
        raise ValueError(
            "Allen model ID mismatch: "
            f"config={expected_model_id}, metadata={observed_model_id}."
        )

    conditions = fit.get("conditions", [])
    if len(conditions) != 1:
        raise ValueError(
            "Expected exactly one conditions entry in fit_parameters.json."
        )

    condition = conditions[0]

    return {
        "source_dir": source_dir,
        "model_id": observed_model_id,
        "specimen_id": int(metadata["specimen_id"]),
        "reconstruction_id": int(metadata["neuron_reconstruction_id"]),
        "model_name": str(metadata["name"]),
        "celsius": float(condition["celsius"]),
        "v_init_mv": float(condition["v_init"]),
        "n_modfiles": len(modfiles),
        "modfiles": [path.name for path in modfiles],
    }


def _prepare_run_directories(
    run_dir: Path,
    *,
    overwrite: bool,
) -> tuple[Path, Path, Path, Path]:
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Run directory already exists: {run_dir}. "
                "Use overwrite=True only for an intentional rebuild."
            )
        shutil.rmtree(run_dir)

    network_dir = run_dir / "network"
    components = run_dir / "components"
    biophys_dir = components / "biophysical_neuron_models"
    morphology_dir = components / "morphologies"
    mechanisms_dir = components / "mechanisms"

    for directory in (
        network_dir,
        biophys_dir,
        morphology_dir,
        mechanisms_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    return (
        network_dir,
        biophys_dir,
        morphology_dir,
        mechanisms_dir,
    )


def _build_bionet_environment(
    run_dir: Path,
    network_dir: Path,
    simulation_cfg: Mapping[str, Any],
    clamp_cfg: Mapping[str, Any],
    *,
    v_init_mv: float,
    celsius: float,
) -> Path:
    from bmtk.utils.sim_setup import build_env_bionet

    build_env_bionet(
        base_dir=str(run_dir),
        config_file="config.json",
        network_dir=str(network_dir),
        tstop=float(simulation_cfg["tstop_ms"]),
        dt=float(simulation_cfg["dt_ms"]),
        dL=float(simulation_cfg["dL_um"]),
        spikes_threshold=float(simulation_cfg["spike_threshold_mv"]),
        v_init=float(v_init_mv),
        celsius=float(celsius),
        report_vars=list(simulation_cfg["report_vars"]),
        current_clamp={
            "amp": float(clamp_cfg["amp_na"]),
            "delay": float(clamp_cfg["delay_ms"]),
            "duration": float(clamp_cfg["duration_ms"]),
        },
        include_examples=False,
        compile_mechanisms=False,
        overwrite_config=True,
    )

    return run_dir / "config.json"


def prepare_allen_reference_run(
    project_root: str | Path,
    config_path: str | Path = "configs/simulations/allen_reference.yaml",
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build a one-cell BioNet run using the native Allen morphology."""

    from bmtk.builder.networks import NetworkBuilder

    root = Path(project_root).resolve()
    settings_path = _resolve_project_path(root, config_path)
    settings = _load_yaml(settings_path)

    model_cfg = settings["allen_reference"]
    simulation_cfg = settings["simulation"]
    clamp_cfg = simulation_cfg["current_clamp"]

    audit = validate_allen_reference_assets(root, settings_path)
    source_dir = Path(audit["source_dir"])

    run_dir = _resolve_project_path(
        root,
        model_cfg["run_dir"],
    ).resolve()

    (
        network_dir,
        biophys_dir,
        morphology_dir,
        mechanisms_dir,
    ) = _prepare_run_directories(
        run_dir,
        overwrite=overwrite,
    )

    population_name = str(model_cfg["population_name"])

    network = NetworkBuilder(population_name)
    network.add_nodes(
        N=1,
        cell_name=str(model_cfg["cell_name"]),
        potential="exc",
        model_type="biophysical",
        model_template=str(model_cfg["model_template"]),
        model_processing=str(model_cfg["model_processing"]),
        dynamics_params=str(model_cfg["dynamics_file"]),
        morphology=str(model_cfg["morphology_file"]),
    )
    network.build()
    network.save_nodes(output_dir=str(network_dir))

    config_file = _build_bionet_environment(
        run_dir,
        network_dir,
        simulation_cfg,
        clamp_cfg,
        v_init_mv=float(audit["v_init_mv"]),
        celsius=float(audit["celsius"]),
    )

    shutil.copy2(
        source_dir / str(model_cfg["dynamics_file"]),
        biophys_dir / str(model_cfg["dynamics_file"]),
    )
    shutil.copy2(
        source_dir / str(model_cfg["morphology_file"]),
        morphology_dir / str(model_cfg["morphology_file"]),
    )
    shutil.copytree(
        source_dir / "modfiles",
        mechanisms_dir / "modfiles",
        dirs_exist_ok=True,
    )

    return {
        **audit,
        "run_dir": run_dir,
        "config_file": config_file,
        "network_dir": network_dir,
        "mechanisms_dir": mechanisms_dir,
        "population_name": population_name,
        "current_clamp_amp_na": float(clamp_cfg["amp_na"]),
        "current_clamp_delay_ms": float(clamp_cfg["delay_ms"]),
        "current_clamp_duration_ms": float(clamp_cfg["duration_ms"]),
        "tstop_ms": float(simulation_cfg["tstop_ms"]),
        "dt_ms": float(simulation_cfg["dt_ms"]),
    }


def validate_microns_single_cell_assets(
    project_root: str | Path,
    config_path: str | Path = "configs/simulations/microns_single_cell.yaml",
) -> dict[str, Any]:
    """Resolve one processed MICrONS morphology and the Allen parameter source."""

    root = Path(project_root).resolve()
    settings_path = _resolve_project_path(root, config_path)
    settings = _load_yaml(settings_path)

    cell_cfg = settings["microns_single_cell"]

    model_node_id = int(cell_cfg["model_node_id"])

    final_manifest_path = _resolve_project_path(
        root,
        cell_cfg["final_manifest"],
    )
    morphology_manifest_path = _resolve_project_path(
        root,
        cell_cfg["morphology_manifest"],
    )
    allen_source_dir = _resolve_project_path(
        root,
        cell_cfg["allen_source_dir"],
    )

    final_manifest = pd.read_parquet(final_manifest_path)
    morphology_manifest = pd.read_parquet(morphology_manifest_path)

    selected = final_manifest.loc[
        final_manifest["model_node_id"].eq(model_node_id)
    ]
    morphology_row = morphology_manifest.loc[
        morphology_manifest["model_node_id"].eq(model_node_id)
    ]

    if len(selected) != 1:
        raise ValueError(
            f"Expected one final-manifest row for model_node_id={model_node_id}, "
            f"found {len(selected)}."
        )

    if len(morphology_row) != 1:
        raise ValueError(
            f"Expected one morphology row for model_node_id={model_node_id}, "
            f"found {len(morphology_row)}."
        )

    selected = selected.iloc[0]
    morphology_row = morphology_row.iloc[0]

    morphology_path = _resolve_project_path(
        root,
        str(morphology_row["simulation_morphology"]),
    ).resolve()

    if not morphology_path.is_file():
        raise FileNotFoundError(morphology_path)

    fit_parameters = allen_source_dir / str(cell_cfg["dynamics_file"])
    modfiles_dir = allen_source_dir / "modfiles"

    if not fit_parameters.is_file():
        raise FileNotFoundError(fit_parameters)

    modfiles = sorted(modfiles_dir.glob("*.mod"))
    if not modfiles:
        raise FileNotFoundError(
            f"No Allen .mod files found under {modfiles_dir}."
        )

    fit = json.loads(fit_parameters.read_text(encoding="utf-8"))
    conditions = fit["conditions"][0]

    return {
        "model_node_id": model_node_id,
        "nucleus_id": int(selected["nucleus_id"]),
        "pt_root_id": int(selected["pt_root_id"]),
        "microns_mtype": str(selected["microns_mtype"]),
        "morphology_path": morphology_path,
        "morphology_name": morphology_path.name,
        "allen_source_dir": allen_source_dir,
        "fit_parameters": fit_parameters,
        "celsius": float(conditions["celsius"]),
        "v_init_mv": float(conditions["v_init"]),
        "n_modfiles": len(modfiles),
    }


def prepare_microns_single_cell_run(
    project_root: str | Path,
    config_path: str | Path = "configs/simulations/microns_single_cell.yaml",
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build one MICrONS cell with provisional Allen-derived dynamics."""

    from bmtk.builder.networks import NetworkBuilder

    root = Path(project_root).resolve()
    settings_path = _resolve_project_path(root, config_path)
    settings = _load_yaml(settings_path)

    cell_cfg = settings["microns_single_cell"]
    simulation_cfg = settings["simulation"]
    clamp_cfg = simulation_cfg["current_clamp"]

    audit = validate_microns_single_cell_assets(
        root,
        settings_path,
    )

    run_dir = _resolve_project_path(
        root,
        cell_cfg["run_dir"],
    ).resolve()

    (
        network_dir,
        biophys_dir,
        morphology_dir,
        mechanisms_dir,
    ) = _prepare_run_directories(
        run_dir,
        overwrite=overwrite,
    )

    population_name = str(cell_cfg["population_name"])

    network = NetworkBuilder(population_name)
    network.add_nodes(
        N=1,
        model_node_id=int(audit["model_node_id"]),
        nucleus_id=int(audit["nucleus_id"]),
        pt_root_id=int(audit["pt_root_id"]),
        microns_mtype=str(audit["microns_mtype"]),
        model_type="biophysical",
        model_template=str(cell_cfg["model_template"]),
        model_processing=str(cell_cfg["model_processing"]),
        dynamics_params=str(cell_cfg["dynamics_file"]),
        morphology=str(audit["morphology_name"]),
        recenter=0,
    )
    network.build()
    network.save_nodes(output_dir=str(network_dir))

    config_file = _build_bionet_environment(
        run_dir,
        network_dir,
        simulation_cfg,
        clamp_cfg,
        v_init_mv=float(audit["v_init_mv"]),
        celsius=float(audit["celsius"]),
    )

    shutil.copy2(
        Path(audit["fit_parameters"]),
        biophys_dir / str(cell_cfg["dynamics_file"]),
    )
    shutil.copy2(
        Path(audit["morphology_path"]),
        morphology_dir / str(audit["morphology_name"]),
    )
    shutil.copytree(
        Path(audit["allen_source_dir"]) / "modfiles",
        mechanisms_dir / "modfiles",
        dirs_exist_ok=True,
    )

    return {
        **audit,
        "run_dir": run_dir,
        "config_file": config_file,
        "network_dir": network_dir,
        "mechanisms_dir": mechanisms_dir,
        "population_name": population_name,
        "current_clamp_amp_na": float(clamp_cfg["amp_na"]),
        "current_clamp_delay_ms": float(clamp_cfg["delay_ms"]),
        "current_clamp_duration_ms": float(clamp_cfg["duration_ms"]),
        "tstop_ms": float(simulation_cfg["tstop_ms"]),
        "dt_ms": float(simulation_cfg["dt_ms"]),
        "model_processing": str(cell_cfg["model_processing"]),
    }


def compile_run_mechanisms(
    run_info: Mapping[str, Any],
) -> Path:
    """Compile NMODL mechanisms inside one simulation run."""

    mechanisms_dir = Path(run_info["mechanisms_dir"]).resolve()

    nrnivmodl = shutil.which("nrnivmodl")
    if nrnivmodl is None:
        raise RuntimeError(
            "nrnivmodl is not available in the active environment."
        )

    result = subprocess.run(
        [nrnivmodl, "modfiles"],
        cwd=mechanisms_dir,
        text=True,
        capture_output=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "NEURON mechanism compilation failed.\n\n"
            f"STDOUT:\n{result.stdout}\n\n"
            f"STDERR:\n{result.stderr}"
        )

    libraries = list(
        (mechanisms_dir / "x86_64").rglob("libnrnmech.so")
    )

    if not libraries:
        raise FileNotFoundError(
            "nrnivmodl returned successfully but libnrnmech.so was not found."
        )

    libraries.sort(
        key=lambda path: (
            ".libs" in path.parts,
            len(path.parts),
        )
    )

    return libraries[0]


def _run_bionet(
    run_info: Mapping[str, Any],
    *,
    register_microns_processors: bool,
) -> Path:
    if register_microns_processors:
        import microns20.cell_processors  # noqa: F401

    from bmtk.simulator import bionet

    config_file = Path(run_info["config_file"]).resolve()
    if not config_file.is_file():
        raise FileNotFoundError(config_file)

    conf = bionet.Config.from_json(str(config_file))
    conf.build_env()

    network = bionet.BioNetwork.from_config(conf)
    simulator = bionet.BioSimulator.from_config(
        conf,
        network=network,
    )
    simulator.run()

    output_dir = Path(run_info["run_dir"]) / "output"
    if not output_dir.is_dir():
        raise RuntimeError(
            "BioNet completed without creating the expected output directory."
        )

    return output_dir


def run_allen_reference(
    run_info: Mapping[str, Any],
) -> Path:
    """Run the native Allen reference simulation."""

    return _run_bionet(
        run_info,
        register_microns_processors=False,
    )


def run_microns_single_cell(
    run_info: Mapping[str, Any],
) -> Path:
    """Run one processed MICrONS morphology with provisional Allen dynamics."""

    return _run_bionet(
        run_info,
        register_microns_processors=True,
    )


def _spike_file_is_empty(run_info: Mapping[str, Any]) -> bool:
    spike_path = Path(run_info["run_dir"]) / "output" / "spikes.h5"

    if not spike_path.is_file():
        return True

    with h5py.File(spike_path, "r") as handle:
        timestamp_datasets: list[h5py.Dataset] = []

        def visitor(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset) and name.endswith("/timestamps"):
                timestamp_datasets.append(obj)

        handle.visititems(visitor)

        if not timestamp_datasets:
            return True

        return all(dataset.size == 0 for dataset in timestamp_datasets)


def simulation_spikes(
    run_info: Mapping[str, Any],
) -> pd.DataFrame:
    """Return a spike table without failing when BMTK records zero spikes."""

    if _spike_file_is_empty(run_info):
        return pd.DataFrame(columns=["node_ids", "timestamps"])

    from bmtk.analyzer.spike_trains import to_dataframe

    return to_dataframe(
        config_file=str(Path(run_info["config_file"]).resolve())
    )


def plot_voltage(
    run_info: Mapping[str, Any],
    *,
    node_ids: list[int] | None = None,
):
    """Plot the membrane-potential report from a BioNet run."""

    from bmtk.analyzer.compartment import plot_traces

    return plot_traces(
        config_file=str(Path(run_info["config_file"]).resolve()),
        node_ids=[0] if node_ids is None else node_ids,
        report_name="v_report",
    )


def allen_reference_spikes(
    run_info: Mapping[str, Any],
) -> pd.DataFrame:
    """Backward-compatible alias for the reference notebook."""

    return simulation_spikes(run_info)


def plot_allen_reference_voltage(
    run_info: Mapping[str, Any],
):
    """Backward-compatible alias for the reference notebook."""

    return plot_voltage(run_info)
