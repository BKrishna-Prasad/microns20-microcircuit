"""Independent current-clamp simulation of all selected MICrONS cells.

This module builds a BioNet experiment from the processed MICrONS population
and morphology manifests. It does not alter the structural SONATA package and
does not introduce recurrent or external synapses.

All cells receive the same Allen-derived provisional perisomatic parameter set
and the same somatic current step. Differences in response therefore reflect
the interaction between the shared provisional membrane model and the
different MICrONS morphologies.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import shutil
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from microns20.simulation import (
    compile_run_mechanisms,
    simulation_spikes,
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


def validate_microns20_independent_assets(
    project_root: str | Path,
    config_path: str | Path = "configs/simulations/microns20_independent.yaml",
) -> dict[str, Any]:
    """Validate the 20-cell population and resolve all simulation assets."""

    root = Path(project_root).resolve()
    settings_path = _resolve_project_path(root, config_path)
    settings = _load_yaml(settings_path)

    experiment = settings["microns20_independent"]

    final_manifest_path = _resolve_project_path(
        root,
        experiment["final_manifest"],
    )
    morphology_manifest_path = _resolve_project_path(
        root,
        experiment["morphology_manifest"],
    )
    allen_source_dir = _resolve_project_path(
        root,
        experiment["allen_source_dir"],
    )

    final_manifest = pd.read_parquet(final_manifest_path)
    morphology_manifest = pd.read_parquet(morphology_manifest_path)

    required_manifest = {
        "model_node_id",
        "nucleus_id",
        "pt_root_id",
        "microns_mtype",
    }
    required_morphology = {
        "model_node_id",
        "nucleus_id",
        "pt_root_id",
        "simulation_morphology",
    }

    missing_manifest = required_manifest - set(final_manifest.columns)
    missing_morphology = required_morphology - set(morphology_manifest.columns)

    if missing_manifest:
        raise ValueError(
            f"Final manifest is missing columns: {sorted(missing_manifest)}"
        )

    if missing_morphology:
        raise ValueError(
            f"Morphology manifest is missing columns: {sorted(missing_morphology)}"
        )

    expected_n = int(experiment["n_cells"])

    if len(final_manifest) != expected_n:
        raise ValueError(
            f"Expected {expected_n} selected neurons, found {len(final_manifest)}."
        )

    expected_ids = list(range(expected_n))
    observed_ids = (
        final_manifest
        .sort_values("model_node_id")["model_node_id"]
        .astype(int)
        .tolist()
    )

    if observed_ids != expected_ids:
        raise ValueError(
            "Expected deterministic model_node_id values 0..n_cells-1."
        )

    identity_columns = [
        "model_node_id",
        "nucleus_id",
        "pt_root_id",
    ]

    compared = (
        final_manifest[identity_columns]
        .merge(
            morphology_manifest[identity_columns],
            on=identity_columns,
            how="outer",
            indicator=True,
        )
    )

    if not compared["_merge"].eq("both").all():
        raise ValueError(
            "Final population and morphology manifest identities do not match."
        )

    population = (
        final_manifest[
            [
                "model_node_id",
                "nucleus_id",
                "pt_root_id",
                "microns_mtype",
            ]
        ]
        .merge(
            morphology_manifest[
                [
                    "model_node_id",
                    "simulation_morphology",
                ]
            ],
            on="model_node_id",
            how="left",
            validate="one_to_one",
        )
        .sort_values("model_node_id")
        .reset_index(drop=True)
    )

    morphology_paths = []
    for relative in population["simulation_morphology"]:
        path = _resolve_project_path(root, str(relative)).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        morphology_paths.append(path)

    population["morphology_path"] = morphology_paths
    population["morphology_name"] = [
        path.name for path in morphology_paths
    ]

    dynamics_file = allen_source_dir / str(experiment["dynamics_file"])
    modfiles_dir = allen_source_dir / "modfiles"

    if not dynamics_file.is_file():
        raise FileNotFoundError(dynamics_file)

    modfiles = sorted(modfiles_dir.glob("*.mod"))
    if not modfiles:
        raise FileNotFoundError(
            f"No Allen .mod files found under {modfiles_dir}."
        )

    fit = json.loads(dynamics_file.read_text(encoding="utf-8"))
    conditions = fit.get("conditions", [])

    if len(conditions) != 1:
        raise ValueError(
            "Expected exactly one Allen conditions entry."
        )

    condition = conditions[0]

    return {
        "population": population,
        "n_cells": len(population),
        "n_l4a": int(population["microns_mtype"].eq("L4a").sum()),
        "n_l4b": int(population["microns_mtype"].eq("L4b").sum()),
        "allen_source_dir": allen_source_dir,
        "dynamics_file": dynamics_file,
        "n_modfiles": len(modfiles),
        "celsius": float(condition["celsius"]),
        "v_init_mv": float(condition["v_init"]),
    }


def prepare_microns20_independent_run(
    project_root: str | Path,
    config_path: str | Path = "configs/simulations/microns20_independent.yaml",
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build a 20-cell BioNet experiment with no inter-cell edges."""

    from bmtk.builder.networks import NetworkBuilder
    from bmtk.utils.sim_setup import build_env_bionet

    root = Path(project_root).resolve()
    settings_path = _resolve_project_path(root, config_path)
    settings = _load_yaml(settings_path)

    experiment = settings["microns20_independent"]
    simulation_cfg = settings["simulation"]
    clamp_cfg = simulation_cfg["current_clamp"]

    audit = validate_microns20_independent_assets(
        root,
        settings_path,
    )

    population = audit["population"].copy()
    run_dir = _resolve_project_path(
        root,
        experiment["run_dir"],
    ).resolve()

    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Run directory already exists: {run_dir}. "
                "Use overwrite=True only for an intentional rebuild."
            )
        shutil.rmtree(run_dir)

    network_dir = run_dir / "network"
    components_dir = run_dir / "components"
    biophys_dir = components_dir / "biophysical_neuron_models"
    morphology_dir = components_dir / "morphologies"
    mechanisms_dir = components_dir / "mechanisms"

    for directory in (
        network_dir,
        biophys_dir,
        morphology_dir,
        mechanisms_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    population_name = str(experiment["population_name"])
    network = NetworkBuilder(population_name)

    for row in population.itertuples(index=False):
        network.add_nodes(
            N=1,
            model_node_id=int(row.model_node_id),
            nucleus_id=int(row.nucleus_id),
            pt_root_id=int(row.pt_root_id),
            microns_mtype=str(row.microns_mtype),
            model_type="biophysical",
            model_template=str(experiment["model_template"]),
            model_processing=str(experiment["model_processing"]),
            dynamics_params=str(experiment["dynamics_file"]),
            morphology=str(row.morphology_name),
            recenter=0,
        )

    network.build()
    network.save_nodes(output_dir=str(network_dir))

    node_mapping = population[
        [
            "model_node_id",
            "nucleus_id",
            "pt_root_id",
            "microns_mtype",
            "morphology_name",
        ]
    ].copy()
    node_mapping.insert(
        0,
        "bmtk_node_id",
        np.arange(len(node_mapping), dtype=int),
    )

    if not np.array_equal(
        node_mapping["bmtk_node_id"].to_numpy(),
        node_mapping["model_node_id"].to_numpy(),
    ):
        raise RuntimeError(
            "This experiment expects BMTK node IDs to match model_node_id."
        )

    node_mapping.to_csv(
        run_dir / "node_mapping.csv",
        index=False,
    )

    build_env_bionet(
        base_dir=str(run_dir),
        config_file="config.json",
        network_dir=str(network_dir),
        components_dir=str(components_dir),
        tstop=float(simulation_cfg["tstop_ms"]),
        dt=float(simulation_cfg["dt_ms"]),
        dL=float(simulation_cfg["dL_um"]),
        spikes_threshold=float(simulation_cfg["spike_threshold_mv"]),
        v_init=float(audit["v_init_mv"]),
        celsius=float(audit["celsius"]),
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

    shutil.copy2(
        Path(audit["dynamics_file"]),
        biophys_dir / str(experiment["dynamics_file"]),
    )

    for row in population.itertuples(index=False):
        shutil.copy2(
            Path(row.morphology_path),
            morphology_dir / str(row.morphology_name),
        )

    shutil.copytree(
        Path(audit["allen_source_dir"]) / "modfiles",
        mechanisms_dir / "modfiles",
        dirs_exist_ok=True,
    )

    return {
        "run_dir": run_dir,
        "config_file": run_dir / "config.json",
        "network_dir": network_dir,
        "components_dir": components_dir,
        "mechanisms_dir": mechanisms_dir,
        "population_name": population_name,
        "node_mapping": node_mapping,
        "n_cells": int(audit["n_cells"]),
        "n_l4a": int(audit["n_l4a"]),
        "n_l4b": int(audit["n_l4b"]),
        "celsius": float(audit["celsius"]),
        "v_init_mv": float(audit["v_init_mv"]),
        "current_clamp_amp_na": float(clamp_cfg["amp_na"]),
        "current_clamp_delay_ms": float(clamp_cfg["delay_ms"]),
        "current_clamp_duration_ms": float(clamp_cfg["duration_ms"]),
        "tstop_ms": float(simulation_cfg["tstop_ms"]),
        "dt_ms": float(simulation_cfg["dt_ms"]),
        "model_processing": str(experiment["model_processing"]),
        "physiology_status": str(experiment["physiology_status"]),
    }


def compile_microns20_independent_mechanisms(
    run_info: Mapping[str, Any],
) -> Path:
    """Compile the Allen NMODL mechanisms copied into this run."""

    return compile_run_mechanisms(run_info)


def run_microns20_independent(
    run_info: Mapping[str, Any],
) -> Path:
    """Run all 20 MICrONS cells independently in a single BioNet simulation."""

    import microns20.cell_processors  # noqa: F401
    from bmtk.simulator import bionet

    config_file = Path(run_info["config_file"]).resolve()

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
            "BioNet completed without producing the expected output directory."
        )

    return output_dir


def _find_voltage_report(
    run_info: Mapping[str, Any],
) -> Path:
    output_dir = Path(run_info["run_dir"]) / "output"

    preferred = output_dir / "v_report.h5"
    if preferred.is_file():
        return preferred

    candidates = sorted(output_dir.glob("*v*.h5"))
    candidates = [
        path
        for path in candidates
        if path.name != "spikes.h5"
    ]

    if len(candidates) != 1:
        raise FileNotFoundError(
            "Could not uniquely resolve the voltage report under "
            f"{output_dir}. Candidates: {[path.name for path in candidates]}"
        )

    return candidates[0]


def load_population_voltage(
    run_info: Mapping[str, Any],
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load one somatic voltage trace per node from the SONATA report."""

    report_path = _find_voltage_report(run_info)
    population_name = str(run_info["population_name"])

    with h5py.File(report_path, "r") as handle:
        report_group = handle[f"report/{population_name}"]
        data = np.asarray(report_group["data"], dtype=float)
        mapping = report_group["mapping"]

        node_ids = np.asarray(
            mapping["node_ids"],
            dtype=int,
        )
        pointer_name = (
            "index_pointers"
            if "index_pointers" in mapping
            else "index_pointer"
        )
        index_pointers = np.asarray(
            mapping[pointer_name],
            dtype=int,
        )
        time_description = np.asarray(
            mapping["time"],
            dtype=float,
        )

    if len(time_description) == 3:
        start_ms, stop_ms, dt_ms = time_description
        times = np.arange(
            start_ms,
            stop_ms,
            dt_ms,
            dtype=float,
        )

        if len(times) != data.shape[0]:
            times = start_ms + np.arange(data.shape[0]) * dt_ms
    else:
        times = time_description

    traces: dict[int, np.ndarray] = {}

    for i, node_id in enumerate(node_ids):
        start = int(index_pointers[i])
        stop = int(index_pointers[i + 1])

        node_data = data[:, start:stop]

        if node_data.ndim != 2 or node_data.shape[1] < 1:
            raise ValueError(
                f"Voltage report contains no values for node {node_id}."
            )

        if node_data.shape[1] != 1:
            raise ValueError(
                "Expected one reported somatic element per node, "
                f"but node {node_id} has {node_data.shape[1]} elements."
            )

        traces[int(node_id)] = node_data[:, 0]

    voltage = pd.DataFrame(
        {
            node_id: traces[node_id]
            for node_id in sorted(traces)
        },
        index=times,
    )
    voltage.index.name = "time_ms"
    voltage.columns.name = "bmtk_node_id"

    return times, voltage


def summarize_microns20_independent(
    run_info: Mapping[str, Any],
) -> pd.DataFrame:
    """Create one simulation-QC row per MICrONS neuron."""

    spikes = simulation_spikes(run_info)
    _, voltage = load_population_voltage(run_info)

    mapping = pd.DataFrame(run_info["node_mapping"]).copy()

    delay = float(run_info["current_clamp_delay_ms"])
    duration = float(run_info["current_clamp_duration_ms"])
    stimulus_end = delay + duration

    baseline_mask = (
        (voltage.index >= max(0.0, delay - 100.0))
        & (voltage.index < delay)
    )
    stimulus_mask = (
        (voltage.index >= delay)
        & (voltage.index < stimulus_end)
    )

    if not baseline_mask.any():
        raise ValueError("No baseline voltage samples were found.")

    if not stimulus_mask.any():
        raise ValueError("No stimulus voltage samples were found.")

    if spikes.empty:
        spike_counts = pd.Series(dtype=int)
        first_spikes = pd.Series(dtype=float)
    else:
        spikes = spikes.copy()
        spikes["node_ids"] = spikes["node_ids"].astype(int)
        spikes["timestamps"] = spikes["timestamps"].astype(float)

        stimulus_spikes = spikes.loc[
            spikes["timestamps"].ge(delay)
            & spikes["timestamps"].lt(stimulus_end)
        ]

        spike_counts = stimulus_spikes.groupby("node_ids").size()
        first_spikes = stimulus_spikes.groupby("node_ids")[
            "timestamps"
        ].min()

    rows = []

    for row in mapping.itertuples(index=False):
        node_id = int(row.bmtk_node_id)

        if node_id not in voltage.columns:
            raise ValueError(
                f"No voltage trace exists for BMTK node {node_id}."
            )

        trace = voltage[node_id]

        baseline = trace.loc[baseline_mask]
        during = trace.loc[stimulus_mask]

        n_spikes = int(spike_counts.get(node_id, 0))
        first_spike_ms = first_spikes.get(node_id, np.nan)

        rows.append(
            {
                "bmtk_node_id": node_id,
                "model_node_id": int(row.model_node_id),
                "nucleus_id": int(row.nucleus_id),
                "pt_root_id": int(row.pt_root_id),
                "microns_mtype": str(row.microns_mtype),
                "simulation_success": bool(
                    np.isfinite(trace.to_numpy()).all()
                ),
                "baseline_vm_mean_mv": float(baseline.mean()),
                "stimulus_vm_mean_mv": float(during.mean()),
                "peak_vm_mv": float(trace.max()),
                "minimum_vm_mv": float(trace.min()),
                "spike_count": n_spikes,
                "first_spike_time_ms": (
                    float(first_spike_ms)
                    if np.isfinite(first_spike_ms)
                    else np.nan
                ),
                "first_spike_latency_ms": (
                    float(first_spike_ms - delay)
                    if np.isfinite(first_spike_ms)
                    else np.nan
                ),
                "stimulus_firing_rate_hz": (
                    float(n_spikes / (duration / 1000.0))
                ),
            }
        )

    summary = pd.DataFrame(rows).sort_values(
        "model_node_id"
    ).reset_index(drop=True)

    if len(summary) != int(run_info["n_cells"]):
        raise RuntimeError(
            "Simulation summary does not contain one row per selected neuron."
        )

    return summary


def save_microns20_independent_summary(
    run_info: Mapping[str, Any],
    summary: pd.DataFrame,
) -> Path:
    """Save the experiment summary beside the generated run output."""

    path = Path(run_info["run_dir"]) / "simulation_summary.csv"
    summary.to_csv(path, index=False)
    return path


def plot_population_voltage_traces(
    run_info: Mapping[str, Any],
    *,
    offset_mv: float = 25.0,
):
    """Plot all somatic voltage traces with a vertical display offset."""

    _, voltage = load_population_voltage(run_info)
    mapping = (
        pd.DataFrame(run_info["node_mapping"])
        .set_index("bmtk_node_id")
    )

    fig, ax = plt.subplots(figsize=(12, 10))

    for order, node_id in enumerate(voltage.columns):
        label = (
            f"M{int(mapping.loc[node_id, 'model_node_id'])} "
            f"({mapping.loc[node_id, 'microns_mtype']})"
        )
        ax.plot(
            voltage.index,
            voltage[node_id] + order * offset_mv,
            linewidth=0.8,
            label=label,
        )

    delay = float(run_info["current_clamp_delay_ms"])
    end = delay + float(run_info["current_clamp_duration_ms"])

    ax.axvline(delay, linestyle="--", linewidth=0.8)
    ax.axvline(end, linestyle="--", linewidth=0.8)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(f"Membrane voltage + {offset_mv:g} mV display offsets")
    ax.set_title(
        "MICrONS20 independent current-clamp responses "
        f"({run_info['current_clamp_amp_na']:.2f} nA)"
    )

    ax.legend(
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        fontsize=8,
    )

    fig.tight_layout()
    return fig


def plot_population_spike_raster(
    run_info: Mapping[str, Any],
):
    """Plot recorded spikes for the 20 independent MICrONS cells."""

    spikes = simulation_spikes(run_info)

    fig, ax = plt.subplots(figsize=(11, 6))

    if not spikes.empty:
        ax.scatter(
            spikes["timestamps"],
            spikes["node_ids"],
            s=12,
        )

    delay = float(run_info["current_clamp_delay_ms"])
    end = delay + float(run_info["current_clamp_duration_ms"])

    ax.axvline(delay, linestyle="--", linewidth=0.8)
    ax.axvline(end, linestyle="--", linewidth=0.8)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("model_node_id")
    ax.set_yticks(range(int(run_info["n_cells"])))
    ax.set_title(
        "MICrONS20 independent current-clamp spike raster "
        f"({run_info['current_clamp_amp_na']:.2f} nA)"
    )

    fig.tight_layout()
    return fig


def plot_population_spike_counts(
    run_info: Mapping[str, Any],
    summary: pd.DataFrame,
):
    """Plot spike count for each selected MICrONS neuron."""

    ordered = summary.sort_values("model_node_id")

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.bar(
        ordered["model_node_id"],
        ordered["spike_count"],
    )

    ax.set_xlabel("model_node_id")
    ax.set_ylabel("Spikes during current step")
    ax.set_xticks(ordered["model_node_id"])
    ax.set_title(
        "MICrONS20 firing under identical provisional stimulation "
        f"({run_info['current_clamp_amp_na']:.2f} nA)"
    )

    fig.tight_layout()
    return fig
