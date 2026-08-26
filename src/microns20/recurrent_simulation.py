"""Build and analyse the recurrent 20-cell MICrONS simulation.

The structural circuit is read from the validated SONATA output produced by
the CAVE pipeline. The selected node HDF5 file and the intrinsic edge HDF5
file are copied without changing node IDs, synapse IDs, section IDs, or
section positions.

Cellular and synaptic physiology used here are modelling assumptions:
- all cells receive the same Allen-derived perisomatic parameter set;
- recurrent excitatory contacts use one simple NEURON Exp2Syn model.

No outside-selected20 inputs are included in this experiment.
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

from microns20.simulation import compile_run_mechanisms, simulation_spikes


SECTION_TYPE_NAMES = {
    1: "soma",
    2: "axon",
    3: "dendrite",
    4: "apical",
}


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


def _project_population_names(project_root: Path) -> dict[str, str]:
    project_config = _load_yaml(
        project_root / "configs" / "project.yaml"
    )
    selected = str(
        project_config["sonata"]["selected_population"]
    )

    return {
        "selected": selected,
        "intrinsic": f"{selected}_to_{selected}",
    }


def _intrinsic_with_model_ids(
    manifest: pd.DataFrame,
    synapses: pd.DataFrame,
) -> pd.DataFrame:
    root_to_model = (
        manifest[["model_node_id", "pt_root_id"]]
        .drop_duplicates()
        .set_index("pt_root_id")["model_node_id"]
        .astype(int)
        .to_dict()
    )

    mapped = synapses.copy()
    mapped["source_model_node_id"] = (
        mapped["pre_pt_root_id"].map(root_to_model)
    )
    mapped["target_model_node_id"] = (
        mapped["post_pt_root_id"].map(root_to_model)
    )

    if mapped[
        ["source_model_node_id", "target_model_node_id"]
    ].isna().any().any():
        raise ValueError(
            "At least one intrinsic synapse endpoint is not in the "
            "selected population."
        )

    mapped["source_model_node_id"] = (
        mapped["source_model_node_id"].astype(int)
    )
    mapped["target_model_node_id"] = (
        mapped["target_model_node_id"].astype(int)
    )

    return mapped


def select_stimulation_source(
    manifest: pd.DataFrame,
    intrinsic_synapses: pd.DataFrame,
) -> tuple[int, pd.DataFrame]:
    """Choose the neuron with the broadest observed recurrent output.

    Ranking:
    1. most distinct postsynaptic target neurons;
    2. most outgoing anatomical contacts;
    3. lowest model_node_id for deterministic tie-breaking.
    """

    mapped = _intrinsic_with_model_ids(
        manifest,
        intrinsic_synapses,
    )

    summary = (
        mapped
        .groupby("source_model_node_id")
        .agg(
            n_distinct_targets=(
                "target_model_node_id",
                "nunique",
            ),
            n_outgoing_contacts=("id", "nunique"),
        )
        .reset_index()
        .rename(
            columns={
                "source_model_node_id": "model_node_id"
            }
        )
    )

    population = (
        manifest[
            [
                "model_node_id",
                "nucleus_id",
                "pt_root_id",
                "microns_mtype",
            ]
        ]
        .drop_duplicates("model_node_id")
        .merge(
            summary,
            on="model_node_id",
            how="left",
            validate="one_to_one",
        )
    )

    population[
        ["n_distinct_targets", "n_outgoing_contacts"]
    ] = population[
        ["n_distinct_targets", "n_outgoing_contacts"]
    ].fillna(0).astype(int)

    ranking = (
        population
        .sort_values(
            [
                "n_distinct_targets",
                "n_outgoing_contacts",
                "model_node_id",
            ],
            ascending=[False, False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    ranking.insert(
        0,
        "source_rank",
        np.arange(1, len(ranking) + 1, dtype=int),
    )
    ranking["selected_for_stimulation"] = False
    ranking.loc[0, "selected_for_stimulation"] = True

    return int(ranking.loc[0, "model_node_id"]), ranking


def validate_recurrent_experiment_assets(
    project_root: str | Path,
    config_path: str | Path = (
        "configs/simulations/microns20_recurrent.yaml"
    ),
) -> dict[str, Any]:
    """Validate structural, cell-model, and synapse-model inputs."""

    root = Path(project_root).resolve()
    settings_path = _resolve_project_path(root, config_path)
    settings = _load_yaml(settings_path)
    experiment = settings["microns20_recurrent"]

    names = _project_population_names(root)
    selected_population = names["selected"]
    intrinsic_population = names["intrinsic"]

    structural_root = _resolve_project_path(
        root,
        experiment["structural_sonata_dir"],
    ).resolve()

    manifest_path = _resolve_project_path(
        root,
        experiment["final_manifest"],
    )
    intrinsic_path = _resolve_project_path(
        root,
        experiment["intrinsic_synapses"],
    )
    morphology_manifest_path = _resolve_project_path(
        root,
        experiment["morphology_manifest"],
    )

    manifest = pd.read_parquet(manifest_path)
    intrinsic = pd.read_parquet(intrinsic_path)
    morphology_manifest = pd.read_parquet(
        morphology_manifest_path
    )

    if len(manifest) != int(experiment["n_cells"]):
        raise ValueError(
            "Selected population size does not match the simulation config."
        )

    expected_ids = list(range(len(manifest)))
    observed_ids = (
        manifest
        .sort_values("model_node_id")["model_node_id"]
        .astype(int)
        .tolist()
    )
    if observed_ids != expected_ids:
        raise ValueError(
            "Expected model_node_id values to be contiguous from zero."
        )

    mapped_intrinsic = _intrinsic_with_model_ids(
        manifest,
        intrinsic,
    )

    if mapped_intrinsic["id"].duplicated().any():
        raise ValueError(
            "Intrinsic CAVE synapse IDs are not unique."
        )

    directed_pairs = (
        mapped_intrinsic[
            ["source_model_node_id", "target_model_node_id"]
        ]
        .drop_duplicates()
    )

    selected_nodes = (
        structural_root
        / "network"
        / "nodes"
        / f"{selected_population}_nodes.h5"
    )
    selected_types = (
        structural_root
        / "network"
        / "nodes"
        / f"{selected_population}_node_types.csv"
    )
    intrinsic_edges = (
        structural_root
        / "network"
        / "edges"
        / f"{intrinsic_population}_edges.h5"
    )
    intrinsic_types = (
        structural_root
        / "network"
        / "edges"
        / f"{intrinsic_population}_edge_types.csv"
    )
    structural_morphologies = (
        structural_root
        / "components"
        / "morphologies"
    )

    required_structural = [
        selected_nodes,
        selected_types,
        intrinsic_edges,
        intrinsic_types,
    ]

    missing = [
        path
        for path in required_structural
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Structural SONATA files are missing: "
            f"{[str(path) for path in missing]}"
        )

    if not structural_morphologies.is_dir():
        raise FileNotFoundError(structural_morphologies)

    with h5py.File(selected_nodes, "r") as handle:
        population = handle[
            f"nodes/{selected_population}"
        ]
        if len(population["node_type_id"]) != len(manifest):
            raise ValueError(
                "Structural SONATA node count does not match the manifest."
            )

    with h5py.File(intrinsic_edges, "r") as handle:
        population = handle[
            f"edges/{intrinsic_population}"
        ]
        n_edges = len(population["edge_type_id"])
        group_index = np.asarray(
            population["edge_group_index"],
            dtype=int,
        )
        group = population["0"]
        h5_synapse_ids = np.asarray(
            group["synapse_id"],
            dtype=np.int64,
        )[group_index]
        h5_section_id = np.asarray(
            group["afferent_section_id"],
            dtype=np.int64,
        )[group_index]
        h5_section_pos = np.asarray(
            group["afferent_section_pos"],
            dtype=float,
        )[group_index]
        h5_section_type = np.asarray(
            group["afferent_section_type"],
            dtype=np.int64,
        )[group_index]

    if n_edges != len(intrinsic):
        raise ValueError(
            "Structural SONATA intrinsic edge count does not match "
            "the intrinsic synapse table."
        )

    if not np.array_equal(
        np.sort(h5_synapse_ids),
        np.sort(intrinsic["id"].to_numpy(dtype=np.int64)),
    ):
        raise ValueError(
            "Structural SONATA synapse IDs do not match the "
            "intrinsic synapse table."
        )

    if (
        not np.isfinite(h5_section_pos).all()
        or (h5_section_pos < 0).any()
        or (h5_section_pos > 1).any()
    ):
        raise ValueError(
            "Structural SONATA contains invalid afferent_section_pos."
        )

    synapse_parameter_path = _resolve_project_path(
        root,
        experiment["synapse"]["parameter_file"],
    ).resolve()

    if not synapse_parameter_path.is_file():
        raise FileNotFoundError(synapse_parameter_path)

    synapse_parameters = json.loads(
        synapse_parameter_path.read_text(encoding="utf-8")
    )

    required_synapse_parameters = {
        "level_of_detail",
        "tau1",
        "tau2",
        "erev",
    }
    missing_synapse_parameters = (
        required_synapse_parameters
        - set(synapse_parameters)
    )

    if missing_synapse_parameters:
        raise ValueError(
            "Synapse parameter file is missing: "
            f"{sorted(missing_synapse_parameters)}"
        )

    if (
        float(synapse_parameters["tau1"]) <= 0
        or float(synapse_parameters["tau2"]) <= 0
        or float(synapse_parameters["tau1"])
        >= float(synapse_parameters["tau2"])
    ):
        raise ValueError(
            "Exp2Syn requires 0 < tau1 < tau2."
        )

    allen_source_dir = _resolve_project_path(
        root,
        experiment["allen_source_dir"],
    ).resolve()

    dynamics_file = (
        allen_source_dir
        / str(experiment["cell_dynamics_file"])
    )
    modfiles_dir = allen_source_dir / "modfiles"

    if not dynamics_file.is_file():
        raise FileNotFoundError(dynamics_file)

    modfiles = sorted(modfiles_dir.glob("*.mod"))
    if not modfiles:
        raise FileNotFoundError(
            f"No Allen .mod files found under {modfiles_dir}."
        )

    fit = json.loads(
        dynamics_file.read_text(encoding="utf-8")
    )
    conditions = fit.get("conditions", [])
    if len(conditions) != 1:
        raise ValueError(
            "Expected exactly one Allen conditions entry."
        )

    source_id, source_ranking = (
        select_stimulation_source(
            manifest,
            intrinsic,
        )
    )

    direct_targets = (
        mapped_intrinsic.loc[
            mapped_intrinsic[
                "source_model_node_id"
            ].eq(source_id)
        ]
        .groupby("target_model_node_id")
        .agg(
            n_contacts_from_source=("id", "nunique")
        )
        .reset_index()
        .rename(
            columns={
                "target_model_node_id": "model_node_id"
            }
        )
        .sort_values("model_node_id")
        .reset_index(drop=True)
    )

    section_summary = (
        pd.DataFrame(
            {
                "afferent_section_type": h5_section_type,
                "afferent_section_id": h5_section_id,
            }
        )
        .assign(
            target_compartment=lambda frame: (
                frame["afferent_section_type"]
                .map(SECTION_TYPE_NAMES)
                .fillna("unknown")
            )
        )
        .groupby(
            [
                "afferent_section_type",
                "target_compartment",
            ],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "n_contacts"})
    )

    condition = conditions[0]

    return {
        "manifest": manifest,
        "intrinsic_synapses": mapped_intrinsic,
        "morphology_manifest": morphology_manifest,
        "selected_population": selected_population,
        "intrinsic_population": intrinsic_population,
        "selected_nodes": selected_nodes,
        "selected_types": selected_types,
        "intrinsic_edges": intrinsic_edges,
        "intrinsic_types": intrinsic_types,
        "structural_morphologies": structural_morphologies,
        "n_cells": len(manifest),
        "n_intrinsic_contacts": len(intrinsic),
        "n_directed_pairs": len(directed_pairs),
        "source_model_node_id": source_id,
        "source_ranking": source_ranking,
        "direct_targets": direct_targets,
        "section_summary": section_summary,
        "n_axon_target_contacts": int(
            np.count_nonzero(h5_section_type == 2)
        ),
        "allen_source_dir": allen_source_dir,
        "cell_dynamics_file": dynamics_file,
        "n_modfiles": len(modfiles),
        "celsius": float(condition["celsius"]),
        "v_init_mv": float(condition["v_init"]),
        "synapse_parameter_path": synapse_parameter_path,
        "synapse_parameters": synapse_parameters,
    }


def _write_space_delimited_types(
    dataframe: pd.DataFrame,
    path: Path,
) -> None:
    """Write a BMTK-compatible SONATA type table."""

    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(
        path,
        sep=" ",
        index=False,
    )


def _add_bmtk_edge_index_compatibility(
    edge_path: Path,
    population_name: str,
) -> list[str]:
    """Add BMTK 1.2-compatible aliases to a run-local SONATA edge file.

    The current SONATA specification names the node-to-range dataset
    ``node_id_to_ranges``. BMTK 1.2.0 expects ``node_id_to_range`` when
    reading edge indices. The validated structural SONATA file is left
    unchanged; this function operates only on the copy inside a simulation
    run and retains the standards-named dataset alongside the compatibility
    alias.

    No edge ordering, node IDs, synapse IDs, section IDs, section positions,
    or biological attributes are changed.
    """

    changes: list[str] = []

    with h5py.File(edge_path, "r+") as handle:
        index_root_path = f"edges/{population_name}/indices"

        if index_root_path not in handle:
            raise KeyError(
                f"Edge population {population_name!r} has no SONATA indices."
            )

        index_root = handle[index_root_path]

        for direction in ("source_to_target", "target_to_source"):
            if direction not in index_root:
                raise KeyError(
                    f"Edge index {index_root_path}/{direction} is missing."
                )

            index_group = index_root[direction]

            standard_name = "node_id_to_ranges"
            bmtk_name = "node_id_to_range"

            if bmtk_name in index_group:
                continue

            if standard_name not in index_group:
                raise KeyError(
                    f"{index_root_path}/{direction} has neither "
                    f"{standard_name!r} nor {bmtk_name!r}."
                )

            standard = np.asarray(
                index_group[standard_name],
                dtype=np.int64,
            )

            alias = index_group.create_dataset(
                bmtk_name,
                data=standard,
                dtype="int64",
            )

            for key, value in index_group[standard_name].attrs.items():
                alias.attrs[key] = value

            changes.append(
                f"{index_root_path}/{direction}/{bmtk_name}"
            )

            if "range_to_edge_id" not in index_group:
                raise KeyError(
                    f"{index_root_path}/{direction}/range_to_edge_id "
                    "is missing."
                )

    return changes


def _adapt_run_local_nodes_for_bmtk(
    node_path: Path,
    original_node_types_path: Path,
    output_node_types_path: Path,
    population_name: str,
    *,
    model_template: str,
    model_processing: str,
    dynamics_params: str,
) -> dict[str, Any]:
    """Move morphology names into the run-local node-types table.

    The validated structural SONATA stores the per-cell ``morphology`` property
    in the HDF5 node group. h5py exposes that variable-length UTF-8 dataset as
    ``bytes`` when read directly, and BMTK 1.2.0 passes the bytes value through
    to the Allen ``Biophys1`` loader. Converting that bytes object with ``str``
    produces a path such as ``b'model_000_....swc'``, which NEURON cannot open.

    For the simulation copy only, this adapter:
    - decodes all morphology names to normal Python strings;
    - removes the HDF5 morphology dataset from the run-local node group;
    - assigns one node_type_id per selected cell;
    - writes the morphology filename as a plain-text node-type property.

    Biological identity fields and node IDs are unchanged. The validated
    structural SONATA package under ``data/processed/sonata`` is not modified.
    """

    with h5py.File(node_path, "r+") as handle:
        population_path = f"nodes/{population_name}"
        if population_path not in handle:
            raise KeyError(
                f"Node population {population_name!r} is missing from {node_path}."
            )

        population = handle[population_path]
        if "0" not in population:
            raise KeyError(
                f"Node population {population_name!r} has no node group '0'."
            )

        group = population["0"]
        if "morphology" not in group:
            raise KeyError(
                "The structural node group does not contain a morphology dataset."
            )

        morphology_dataset = group["morphology"]
        morphology_names = morphology_dataset.asstr()[:].tolist()

        if len(morphology_names) != len(population["node_type_id"]):
            raise ValueError(
                "Morphology count does not match selected node count."
            )

        if any(
            not isinstance(name, str) or not name
            for name in morphology_names
        ):
            raise ValueError(
                "At least one morphology name could not be decoded as text."
            )

        del group["morphology"]

        node_type_ids = np.arange(
            len(morphology_names),
            dtype=np.uint64,
        )
        population["node_type_id"][...] = node_type_ids

    original_types = pd.read_csv(original_node_types_path)
    if len(original_types) != 1:
        raise ValueError(
            "Expected exactly one structural node-type row before "
            "the run-local BMTK adaptation."
        )

    base = original_types.iloc[0].to_dict()
    rows: list[dict[str, Any]] = []

    for node_type_id, morphology_name in enumerate(morphology_names):
        row = dict(base)
        row.update(
            {
                "node_type_id": int(node_type_id),
                "population": population_name,
                "model_type": "biophysical",
                "recenter": 0,
                "model_template": model_template,
                "model_processing": model_processing,
                "dynamics_params": dynamics_params,
                "morphology": morphology_name,
                "scientific_status": (
                    "allen_derived_parameters_not_microns_fitted"
                ),
            }
        )
        rows.append(row)

    run_types = pd.DataFrame(rows)
    _write_space_delimited_types(
        run_types,
        output_node_types_path,
    )

    return {
        "n_nodes_adapted": len(morphology_names),
        "hdf5_morphology_property_removed": True,
        "morphology_property_location": "node_types_table",
        "node_type_ids": node_type_ids.astype(int).tolist(),
        "morphology_names": morphology_names,
    }


def _copy_network_and_components(
    root: Path,
    run_dir: Path,
    experiment: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> tuple[Path, Path]:
    network_dir = run_dir / "network"
    components_dir = run_dir / "components"

    node_model_dir = (
        components_dir / "biophysical_neuron_models"
    )
    morphology_dir = (
        components_dir / "morphologies"
    )
    mechanism_dir = (
        components_dir / "mechanisms"
    )
    synapse_dir = (
        components_dir / "synaptic_models"
    )

    for directory in (
        network_dir,
        node_model_dir,
        morphology_dir,
        mechanism_dir,
        synapse_dir,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    selected_population = str(
        audit["selected_population"]
    )
    intrinsic_population = str(
        audit["intrinsic_population"]
    )

    run_nodes = (
        network_dir
        / f"{selected_population}_nodes.h5"
    )
    run_node_types = (
        network_dir
        / f"{selected_population}_node_types.csv"
    )
    run_edges = (
        network_dir
        / f"{intrinsic_population}_edges.h5"
    )
    run_edge_types = (
        network_dir
        / f"{intrinsic_population}_edge_types.csv"
    )

    shutil.copy2(
        Path(audit["selected_nodes"]),
        run_nodes,
    )
    shutil.copy2(
        Path(audit["intrinsic_edges"]),
        run_edges,
    )

    bmtk_edge_index_aliases = _add_bmtk_edge_index_compatibility(
        run_edges,
        intrinsic_population,
    )

    compatibility_path = run_dir / "bmtk_edge_index_compatibility.json"
    compatibility_path.write_text(
        json.dumps(
            {
                "bmtk_version_target": "1.2.x",
                "structural_sonata_modified": False,
                "run_local_edge_file": str(run_edges.relative_to(run_dir)),
                "aliases_added": bmtk_edge_index_aliases,
                "reason": (
                    "BMTK 1.2.x reads node_id_to_range while the "
                    "SONATA edge-index specification uses node_id_to_ranges."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    node_compatibility = _adapt_run_local_nodes_for_bmtk(
        run_nodes,
        Path(audit["selected_types"]),
        run_node_types,
        selected_population,
        model_template=str(
            experiment["cell_model_template"]
        ),
        model_processing=str(
            experiment["cell_model_processing"]
        ),
        dynamics_params=str(
            experiment["cell_dynamics_file"]
        ),
    )

    node_compatibility_path = (
        run_dir / "bmtk_node_morphology_compatibility.json"
    )
    node_compatibility_path.write_text(
        json.dumps(
            {
                "bmtk_version_target": "1.2.x",
                "structural_sonata_modified": False,
                "run_local_node_file": str(
                    run_nodes.relative_to(run_dir)
                ),
                **node_compatibility,
                "reason": (
                    "BMTK 1.2.x passes the HDF5 morphology property as "
                    "bytes to the Allen Biophys1 loader. The run-local "
                    "adapter places decoded morphology names in the "
                    "node-types table instead."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    edge_types = pd.read_csv(
        audit["intrinsic_types"]
    )
    if len(edge_types) != 1:
        raise ValueError(
            "Expected one intrinsic edge type."
        )

    edge_types = edge_types.copy()
    edge_types.loc[:, "model_template"] = str(
        experiment["synapse"]["model_template"]
    )
    edge_types.loc[:, "dynamics_params"] = (
        Path(
            str(
                experiment["synapse"][
                    "parameter_file"
                ]
            )
        ).name
    )
    edge_types.loc[:, "syn_weight"] = float(
        experiment["synapse"]["weight_us"]
    )
    edge_types.loc[:, "delay"] = float(
        experiment["synapse"]["delay_ms"]
    )
    edge_types.loc[:, "scientific_status"] = (
        "reference_excitatory_synapse_not_microns_measured"
    )

    _write_space_delimited_types(
        edge_types,
        run_edge_types,
    )

    shutil.copy2(
        Path(audit["cell_dynamics_file"]),
        node_model_dir
        / Path(
            audit["cell_dynamics_file"]
        ).name,
    )

    for morphology in sorted(
        Path(
            audit["structural_morphologies"]
        ).glob("*.swc")
    ):
        shutil.copy2(
            morphology,
            morphology_dir / morphology.name,
        )

    if len(
        list(morphology_dir.glob("*.swc"))
    ) != int(audit["n_cells"]):
        raise RuntimeError(
            "The recurrent run does not contain one "
            "morphology per selected neuron."
        )

    shutil.copytree(
        Path(audit["allen_source_dir"])
        / "modfiles",
        mechanism_dir / "modfiles",
        dirs_exist_ok=True,
    )

    shutil.copy2(
        Path(audit["synapse_parameter_path"]),
        synapse_dir
        / Path(
            audit["synapse_parameter_path"]
        ).name,
    )

    return network_dir, components_dir


def _json_candidates(
    value: Any,
) -> list[str]:
    values: list[str] = []

    if isinstance(value, dict):
        for child in value.values():
            values.extend(
                _json_candidates(child)
            )
    elif isinstance(value, list):
        for child in value:
            values.extend(
                _json_candidates(child)
            )
    elif (
        isinstance(value, str)
        and value.endswith(".json")
    ):
        values.append(value)

    return values


def _resolve_simulation_config(
    run_dir: Path,
    main_config_path: Path,
) -> Path:
    main = json.loads(
        main_config_path.read_text(
            encoding="utf-8"
        )
    )

    if "run" in main and "inputs" in main:
        return main_config_path

    candidates: list[Path] = []

    for value in _json_candidates(main):
        cleaned = (
            value
            .replace("$BASE_DIR", ".")
            .replace("${BASE_DIR}", ".")
        )
        candidate = (
            run_dir / cleaned
        ).resolve()

        if candidate.is_file():
            candidates.append(candidate)

    candidates.extend(
        sorted(
            run_dir.glob("simulation*.json")
        )
    )

    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)

    for candidate in unique:
        try:
            data = json.loads(
                candidate.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            continue

        if "run" in data:
            return candidate

    raise FileNotFoundError(
        "Could not identify the BioNet simulation "
        f"configuration under {run_dir}."
    )


def _restrict_current_clamp_to_source(
    simulation_config_path: Path,
    source_model_node_id: int,
) -> None:
    config = json.loads(
        simulation_config_path.read_text(
            encoding="utf-8"
        )
    )

    inputs = config.get("inputs", {})
    clamp_names = [
        name
        for name, entry in inputs.items()
        if (
            str(entry.get("module", "")).lower()
            == "iclamp"
            or str(
                entry.get(
                    "input_type",
                    "",
                )
            ).lower()
            == "current_clamp"
        )
    ]

    if len(clamp_names) != 1:
        raise ValueError(
            "Expected exactly one current-clamp input, "
            f"found {clamp_names}."
        )

    clamp = inputs[clamp_names[0]]
    clamp["node_set"] = [
        int(source_model_node_id)
    ]
    clamp.pop("gids", None)

    simulation_config_path.write_text(
        json.dumps(
            config,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def prepare_recurrent_run(
    project_root: str | Path,
    config_path: str | Path = (
        "configs/simulations/microns20_recurrent.yaml"
    ),
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Prepare a recurrent BioNet run from the validated structural SONATA."""

    from bmtk.utils.sim_setup import build_env_bionet

    root = Path(project_root).resolve()
    settings_path = _resolve_project_path(
        root,
        config_path,
    )
    settings = _load_yaml(settings_path)
    experiment = settings["microns20_recurrent"]
    simulation = settings["simulation"]
    clamp = simulation["source_current"]

    audit = validate_recurrent_experiment_assets(
        root,
        settings_path,
    )

    run_dir = _resolve_project_path(
        root,
        experiment["run_dir"],
    ).resolve()

    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Run directory already exists: {run_dir}."
            )
        shutil.rmtree(run_dir)

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    network_dir, components_dir = (
        _copy_network_and_components(
            root,
            run_dir,
            experiment,
            audit,
        )
    )

    main_config = run_dir / "config.json"

    build_env_bionet(
        base_dir=str(run_dir),
        config_file="config.json",
        network_dir=str(network_dir),
        components_dir=str(components_dir),
        tstop=float(
            simulation["tstop_ms"]
        ),
        dt=float(
            simulation["dt_ms"]
        ),
        dL=float(
            simulation["dL_um"]
        ),
        spikes_threshold=float(
            simulation[
                "spike_threshold_mv"
            ]
        ),
        v_init=float(
            audit["v_init_mv"]
        ),
        celsius=float(
            audit["celsius"]
        ),
        report_vars=list(
            simulation["report_vars"]
        ),
        current_clamp={
            "amp": float(
                clamp["amp_na"]
            ),
            "delay": float(
                clamp["delay_ms"]
            ),
            "duration": float(
                clamp["duration_ms"]
            ),
        },
        include_examples=False,
        compile_mechanisms=False,
        overwrite_config=True,
    )

    simulation_config = (
        _resolve_simulation_config(
            run_dir,
            main_config,
        )
    )

    _restrict_current_clamp_to_source(
        simulation_config,
        int(
            audit[
                "source_model_node_id"
            ]
        ),
    )

    source_ranking_path = (
        run_dir / "source_selection.csv"
    )
    audit["source_ranking"].to_csv(
        source_ranking_path,
        index=False,
    )

    direct_targets_path = (
        run_dir / "direct_targets.csv"
    )
    audit["direct_targets"].to_csv(
        direct_targets_path,
        index=False,
    )

    section_summary_path = (
        run_dir
        / "intrinsic_target_compartments.csv"
    )
    audit["section_summary"].to_csv(
        section_summary_path,
        index=False,
    )

    run_metadata = {
        "selected_population": str(
            audit["selected_population"]
        ),
        "intrinsic_population": str(
            audit["intrinsic_population"]
        ),
        "n_cells": int(
            audit["n_cells"]
        ),
        "n_intrinsic_contacts": int(
            audit[
                "n_intrinsic_contacts"
            ]
        ),
        "n_directed_pairs": int(
            audit["n_directed_pairs"]
        ),
        "source_model_node_id": int(
            audit[
                "source_model_node_id"
            ]
        ),
        "n_direct_targets": int(
            len(
                audit[
                    "direct_targets"
                ]
            )
        ),
        "n_axon_target_contacts": int(
            audit[
                "n_axon_target_contacts"
            ]
        ),
        "cell_model": {
            "source": "Allen Cell Types",
            "dynamics_file": str(
                experiment[
                    "cell_dynamics_file"
                ]
            ),
            "model_template": str(
                experiment[
                    "cell_model_template"
                ]
            ),
            "model_processing": str(
                experiment[
                    "cell_model_processing"
                ]
            ),
            "status": (
                "Allen-derived parameters; "
                "not fitted to MICrONS physiology"
            ),
        },
        "synapse_model": {
            "model_template": str(
                experiment["synapse"][
                    "model_template"
                ]
            ),
            "parameter_file": str(
                experiment["synapse"][
                    "parameter_file"
                ]
            ),
            "weight_us": float(
                experiment["synapse"][
                    "weight_us"
                ]
            ),
            "delay_ms": float(
                experiment["synapse"][
                    "delay_ms"
                ]
            ),
            "parameters": dict(
                audit[
                    "synapse_parameters"
                ]
            ),
            "status": (
                "Reference excitatory parameters; "
                "not measured for these MICrONS contacts"
            ),
        },
        "stimulation": {
            "node_id": int(
                audit[
                    "source_model_node_id"
                ]
            ),
            "amp_na": float(
                clamp["amp_na"]
            ),
            "delay_ms": float(
                clamp["delay_ms"]
            ),
            "duration_ms": float(
                clamp["duration_ms"]
            ),
        },
        "outside_selected20_inputs_included": False,
        "bmtk_edge_index_compatibility": {
            "structural_sonata_modified": False,
            "run_local_adapter": True,
            "report_file": "bmtk_edge_index_compatibility.json",
        },
        "bmtk_node_morphology_compatibility": {
            "structural_sonata_modified": False,
            "run_local_adapter": True,
            "report_file": "bmtk_node_morphology_compatibility.json",
        },
    }

    metadata_path = (
        run_dir / "run_metadata.json"
    )
    metadata_path.write_text(
        json.dumps(
            run_metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        **{
            key: value
            for key, value in audit.items()
            if key
            not in {
                "manifest",
                "intrinsic_synapses",
                "morphology_manifest",
                "source_ranking",
                "direct_targets",
                "section_summary",
            }
        },
        "manifest": audit["manifest"],
        "intrinsic_synapses": audit[
            "intrinsic_synapses"
        ],
        "source_ranking": audit[
            "source_ranking"
        ],
        "direct_targets": audit[
            "direct_targets"
        ],
        "section_summary": audit[
            "section_summary"
        ],
        "run_dir": run_dir,
        "network_dir": network_dir,
        "components_dir": components_dir,
        "mechanisms_dir": (
            components_dir
            / "mechanisms"
        ),
        "config_file": main_config,
        "simulation_config": (
            simulation_config
        ),
        "source_selection_path": (
            source_ranking_path
        ),
        "direct_targets_path": (
            direct_targets_path
        ),
        "section_summary_path": (
            section_summary_path
        ),
        "run_metadata_path": (
            metadata_path
        ),
        "bmtk_edge_index_compatibility_path": (
            run_dir / "bmtk_edge_index_compatibility.json"
        ),
        "bmtk_node_morphology_compatibility_path": (
            run_dir / "bmtk_node_morphology_compatibility.json"
        ),
        "current_clamp_amp_na": float(
            clamp["amp_na"]
        ),
        "current_clamp_delay_ms": float(
            clamp["delay_ms"]
        ),
        "current_clamp_duration_ms": float(
            clamp["duration_ms"]
        ),
        "post_stimulus_window_ms": float(
            settings["analysis"][
                "post_stimulus_window_ms"
            ]
        ),
        "tstop_ms": float(
            simulation["tstop_ms"]
        ),
        "dt_ms": float(
            simulation["dt_ms"]
        ),
    }


def compile_recurrent_mechanisms(
    run_info: Mapping[str, Any],
) -> Path:
    """Compile the Allen NMODL mechanisms copied into the recurrent run."""

    return compile_run_mechanisms(
        run_info
    )


def run_recurrent_network(
    run_info: Mapping[str, Any],
) -> Path:
    """Run the 20-cell recurrent BioNet simulation."""

    import microns20.cell_processors  # noqa: F401
    from bmtk.simulator import bionet

    config_file = Path(
        run_info["config_file"]
    ).resolve()

    conf = bionet.Config.from_json(
        str(config_file)
    )
    conf.build_env()

    network = bionet.BioNetwork.from_config(
        conf
    )

    simulator = (
        bionet.BioSimulator.from_config(
            conf,
            network=network,
        )
    )
    simulator.run()

    output_dir = (
        Path(run_info["run_dir"])
        / "output"
    )

    if not output_dir.is_dir():
        raise RuntimeError(
            "BioNet completed without creating "
            "the expected output directory."
        )

    return output_dir


def _find_voltage_report(
    run_info: Mapping[str, Any],
) -> Path:
    output_dir = (
        Path(run_info["run_dir"])
        / "output"
    )

    preferred = (
        output_dir / "v_report.h5"
    )
    if preferred.is_file():
        return preferred

    candidates = [
        path
        for path in sorted(
            output_dir.glob("*v*.h5")
        )
        if path.name != "spikes.h5"
    ]

    if len(candidates) != 1:
        raise FileNotFoundError(
            "Could not uniquely resolve the "
            f"voltage report: {candidates}"
        )

    return candidates[0]


def load_recurrent_voltage(
    run_info: Mapping[str, Any],
) -> pd.DataFrame:
    """Load one somatic voltage trace per selected cell."""

    report_path = _find_voltage_report(
        run_info
    )
    population_name = str(
        run_info["selected_population"]
    )

    with h5py.File(
        report_path,
        "r",
    ) as handle:
        report_group = handle[
            f"report/{population_name}"
        ]
        data = np.asarray(
            report_group["data"],
            dtype=float,
        )
        mapping = report_group[
            "mapping"
        ]

        node_ids = np.asarray(
            mapping["node_ids"],
            dtype=int,
        )

        if "index_pointers" in mapping:
            pointer_name = (
                "index_pointers"
            )
        elif "index_pointer" in mapping:
            pointer_name = (
                "index_pointer"
            )
        else:
            raise KeyError(
                "Voltage report mapping has "
                "neither 'index_pointers' nor "
                "'index_pointer'."
            )

        index_pointer = np.asarray(
            mapping[pointer_name],
            dtype=int,
        )
        time_description = np.asarray(
            mapping["time"],
            dtype=float,
        )

    if len(time_description) == 3:
        start_ms, _, dt_ms = (
            time_description
        )
        times = (
            start_ms
            + np.arange(
                data.shape[0],
                dtype=float,
            )
            * dt_ms
        )
    elif len(time_description) == data.shape[0]:
        times = time_description
    else:
        raise ValueError(
            "Unexpected SONATA voltage-report "
            "time description."
        )

    traces: dict[int, np.ndarray] = {}

    for i, node_id in enumerate(
        node_ids
    ):
        start = int(
            index_pointer[i]
        )
        stop = int(
            index_pointer[i + 1]
        )
        node_data = data[
            :,
            start:stop,
        ]

        if node_data.shape[1] != 1:
            raise ValueError(
                "Expected one somatic report "
                f"element for node {node_id}, "
                f"found {node_data.shape[1]}."
            )

        traces[int(node_id)] = (
            node_data[:, 0]
        )

    voltage = pd.DataFrame(
        {
            node_id: traces[node_id]
            for node_id in sorted(traces)
        },
        index=times,
    )
    voltage.index.name = "time_ms"
    voltage.columns.name = (
        "model_node_id"
    )

    return voltage


def summarize_recurrent_run(
    run_info: Mapping[str, Any],
) -> pd.DataFrame:
    """Summarize source, direct-target, and other cell responses."""

    manifest = (
        pd.DataFrame(
            run_info["manifest"]
        )
        .sort_values("model_node_id")
        .reset_index(drop=True)
    )

    source_id = int(
        run_info["source_model_node_id"]
    )

    direct_target_contacts = (
        pd.DataFrame(
            run_info["direct_targets"]
        )
        .set_index("model_node_id")[
            "n_contacts_from_source"
        ]
        .astype(int)
        .to_dict()
    )

    spikes = simulation_spikes(
        run_info
    )
    voltage = load_recurrent_voltage(
        run_info
    )

    delay = float(
        run_info[
            "current_clamp_delay_ms"
        ]
    )
    duration = float(
        run_info[
            "current_clamp_duration_ms"
        ]
    )
    post_window = float(
        run_info[
            "post_stimulus_window_ms"
        ]
    )

    stimulus_end = (
        delay + duration
    )
    analysis_end = min(
        float(
            run_info["tstop_ms"]
        ),
        stimulus_end + post_window,
    )

    baseline_mask = (
        (voltage.index >= max(
            0.0,
            delay - 100.0,
        ))
        & (voltage.index < delay)
    )
    analysis_mask = (
        (voltage.index >= delay)
        & (
            voltage.index
            < analysis_end
        )
    )

    if not baseline_mask.any():
        raise ValueError(
            "No baseline voltage samples "
            "were found."
        )

    if not analysis_mask.any():
        raise ValueError(
            "No network-analysis voltage "
            "samples were found."
        )

    if spikes.empty:
        spike_counts = (
            pd.Series(dtype=int)
        )
        first_spikes = (
            pd.Series(dtype=float)
        )
    else:
        spikes = spikes.copy()
        spikes["node_ids"] = (
            spikes["node_ids"].astype(int)
        )
        spikes["timestamps"] = (
            spikes["timestamps"].astype(float)
        )

        analysis_spikes = spikes.loc[
            spikes["timestamps"].ge(
                delay
            )
            & spikes["timestamps"].lt(
                analysis_end
            )
        ]

        spike_counts = (
            analysis_spikes
            .groupby("node_ids")
            .size()
        )
        first_spikes = (
            analysis_spikes
            .groupby("node_ids")[
                "timestamps"
            ]
            .min()
        )

    rows: list[dict[str, Any]] = []

    for row in manifest.itertuples(
        index=False
    ):
        model_id = int(
            row.model_node_id
        )

        if model_id not in voltage:
            raise ValueError(
                f"No voltage trace for "
                f"model_node_id={model_id}."
            )

        trace = voltage[model_id]
        baseline = trace.loc[
            baseline_mask
        ]
        during = trace.loc[
            analysis_mask
        ]

        if model_id == source_id:
            role = "stimulated_source"
        elif model_id in (
            direct_target_contacts
        ):
            role = "direct_target"
        else:
            role = "other"

        n_spikes = int(
            spike_counts.get(
                model_id,
                0,
            )
        )
        first_spike = first_spikes.get(
            model_id,
            np.nan,
        )

        baseline_mean = float(
            baseline.mean()
        )
        peak_analysis = float(
            during.max()
        )

        rows.append(
            {
                "model_node_id": model_id,
                "nucleus_id": int(
                    row.nucleus_id
                ),
                "microns_mtype": str(
                    row.microns_mtype
                ),
                "network_role": role,
                "n_contacts_from_source": int(
                    direct_target_contacts.get(
                        model_id,
                        0,
                    )
                ),
                "simulation_success": bool(
                    np.isfinite(
                        trace.to_numpy()
                    ).all()
                ),
                "baseline_vm_mean_mv": (
                    baseline_mean
                ),
                "analysis_peak_vm_mv": (
                    peak_analysis
                ),
                "max_depolarization_mv": (
                    peak_analysis
                    - baseline_mean
                ),
                "spike_count": (
                    n_spikes
                ),
                "first_spike_time_ms": (
                    float(first_spike)
                    if np.isfinite(
                        first_spike
                    )
                    else np.nan
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("model_node_id")
        .reset_index(drop=True)
    )


def save_recurrent_summary(
    run_info: Mapping[str, Any],
    summary: pd.DataFrame,
) -> Path:
    """Save the per-cell network-response table."""

    path = (
        Path(run_info["run_dir"])
        / "network_response_summary.csv"
    )
    summary.to_csv(
        path,
        index=False,
    )
    return path


def plot_recurrent_spike_raster(
    run_info: Mapping[str, Any],
):
    """Plot all spikes with the stimulated source marked on the y axis."""

    spikes = simulation_spikes(
        run_info
    )
    source_id = int(
        run_info["source_model_node_id"]
    )

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    if not spikes.empty:
        ax.scatter(
            spikes["timestamps"],
            spikes["node_ids"],
            s=14,
        )

    delay = float(
        run_info[
            "current_clamp_delay_ms"
        ]
    )
    end = (
        delay
        + float(
            run_info[
                "current_clamp_duration_ms"
            ]
        )
    )

    ax.axvline(
        delay,
        linestyle="--",
        linewidth=0.8,
    )
    ax.axvline(
        end,
        linestyle="--",
        linewidth=0.8,
    )

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("model_node_id")
    ax.set_yticks(
        range(
            int(run_info["n_cells"])
        )
    )
    ax.set_title(
        "MICrONS20 recurrent network: "
        f"current applied to model {source_id}"
    )

    fig.tight_layout()
    return fig


def plot_source_and_target_voltage(
    run_info: Mapping[str, Any],
):
    """Plot voltage traces for the source and its direct recurrent targets."""

    voltage = load_recurrent_voltage(
        run_info
    )
    source_id = int(
        run_info["source_model_node_id"]
    )
    targets = sorted(
        pd.DataFrame(
            run_info["direct_targets"]
        )["model_node_id"]
        .astype(int)
        .tolist()
    )

    node_ids = [
        source_id,
        *targets,
    ]

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    offset_mv = 25.0

    for order, node_id in enumerate(
        node_ids
    ):
        role = (
            "source"
            if node_id == source_id
            else "direct target"
        )
        ax.plot(
            voltage.index,
            voltage[node_id]
            + order * offset_mv,
            linewidth=0.9,
            label=(
                f"M{node_id} "
                f"({role})"
            ),
        )

    delay = float(
        run_info[
            "current_clamp_delay_ms"
        ]
    )
    end = (
        delay
        + float(
            run_info[
                "current_clamp_duration_ms"
            ]
        )
    )

    ax.axvline(
        delay,
        linestyle="--",
        linewidth=0.8,
    )
    ax.axvline(
        end,
        linestyle="--",
        linewidth=0.8,
    )

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(
        "Membrane voltage "
        f"+ {offset_mv:g} mV display offsets"
    )
    ax.set_title(
        "Stimulated neuron and its "
        "direct recurrent targets"
    )
    ax.legend(
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        fontsize=8,
    )

    fig.tight_layout()
    return fig
