"""Build a structural SONATA package without invented physiology."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import h5py
import libsonata
import numpy as np
import pandas as pd

from microns20.artifacts import require_columns, sha256_file


def _atomic_h5(path: Path, writer: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".h5", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_csv(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".csv", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        dataframe.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()

def get_sonata_population_names(config: Mapping[str, Any]) -> dict[str, str]:
    """Return SONATA population names from project configuration."""

    selected = str(config["sonata"]["selected_population"])
    external = str(config["sonata"]["external_population"])

    return {
        "selected": selected,
        "external": external,
        "intrinsic_edges": f"{selected}_to_{selected}",
        "external_edges": f"{external}_to_{selected}",
    }

def _write_selected_nodes(
    path: Path,
    manifest: pd.DataFrame,
    morphology_names: list[str],
    population_name: str,
) -> None:
    """Write the selected biological population with explicit SONATA node IDs."""

    ordered = manifest.sort_values("model_node_id").reset_index(drop=True)
    expected_ids = list(range(len(ordered)))
    if ordered["model_node_id"].astype(int).tolist() != expected_ids:
        raise ValueError("SONATA selected node IDs must equal deterministic model IDs.")
    if len(morphology_names) != len(ordered):
        raise ValueError("Morphology-name count does not match selected-node count.")

    def writer(target: Path) -> None:
        with h5py.File(target, "w") as handle:
            population = handle.create_group(f"nodes/{population_name}")
            count = len(ordered)
            population.create_dataset(
                "node_id",
                data=ordered["model_node_id"].to_numpy(dtype="uint64"),
            )
            population.create_dataset(
                "node_type_id",
                data=np.zeros(count, dtype="uint64"),
            )
            population.create_dataset(
                "node_group_id",
                data=np.zeros(count, dtype="uint32"),
            )
            population.create_dataset(
                "node_group_index",
                data=np.arange(count, dtype="uint64"),
            )
            group = population.create_group("0")
            for column in (
                "model_node_id",
                "nucleus_id",
                "pt_root_id",
                "pt_supervoxel_id",
                "session",
                "scan_idx",
            ):
                group.create_dataset(
                    column,
                    data=ordered[column].to_numpy(dtype="int64"),
                )
            for column in ("soma_x_um", "soma_y_um", "soma_z_um"):
                dataset = group.create_dataset(
                    column,
                    data=ordered[column].to_numpy(dtype="float64"),
                )
                dataset.attrs["units"] = "micrometre"
            group.create_dataset(
                "morphology",
                data=np.asarray(
                    morphology_names,
                    dtype=h5py.string_dtype("utf-8"),
                ),
            )

    _atomic_h5(path, writer)


def _write_external_nodes(
    path: Path,
    external_manifest: pd.DataFrame,
    population_name: str,
) -> None:
    """Write one deterministic virtual node for each positive presynaptic root."""

    ordered = external_manifest.sort_values("external_node_id").reset_index(drop=True)
    expected_ids = list(range(len(ordered)))
    if ordered["external_node_id"].astype(int).tolist() != expected_ids:
        raise ValueError("External node IDs must be contiguous from zero.")

    def writer(target: Path) -> None:
        with h5py.File(target, "w") as handle:
            population = handle.create_group(f"nodes/{population_name}")
            count = len(ordered)
            population.create_dataset(
                "node_id",
                data=ordered["external_node_id"].to_numpy(dtype="uint64"),
            )
            population.create_dataset(
                "node_type_id",
                data=np.full(count, 100, dtype="uint64"),
            )
            population.create_dataset(
                "node_group_id",
                data=np.zeros(count, dtype="uint32"),
            )
            population.create_dataset(
                "node_group_index",
                data=np.arange(count, dtype="uint64"),
            )
            group = population.create_group("0")
            group.create_dataset(
                "external_node_id",
                data=ordered["external_node_id"].to_numpy(dtype="int64"),
            )
            group.create_dataset(
                "pt_root_id",
                data=ordered["pt_root_id"].to_numpy(dtype="int64"),
            )

    _atomic_h5(path, writer)


def _edge_arrays(
    synapses: pd.DataFrame,
    source_ids: np.ndarray,
    target_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    """Create one homogeneous SONATA edge-group array per observed synapse."""

    require_columns(
        synapses,
        {
            "id", "size", "afferent_section_id", "afferent_section_pos",
            "afferent_section_type", "afferent_center_x_um",
            "afferent_center_y_um", "afferent_center_z_um",
            "synapse_to_section_distance_um",
            "synapse_to_cave_skeleton_vertex_distance_um",
        },
        "mapped synapses",
    )
    return {
        "source_node_id": np.asarray(source_ids, dtype="uint64"),
        "target_node_id": np.asarray(target_ids, dtype="uint64"),
        "synapse_id": synapses["id"].to_numpy(dtype="int64"),
        "synapse_size": synapses["size"].to_numpy(dtype="float64"),
        "afferent_section_id": synapses["afferent_section_id"].to_numpy(dtype="int64"),
        "afferent_section_pos": synapses["afferent_section_pos"].to_numpy(dtype="float64"),
        "afferent_section_type": synapses["afferent_section_type"].to_numpy(dtype="int64"),
        "afferent_center_x": synapses["afferent_center_x_um"].to_numpy(dtype="float64"),
        "afferent_center_y": synapses["afferent_center_y_um"].to_numpy(dtype="float64"),
        "afferent_center_z": synapses["afferent_center_z_um"].to_numpy(dtype="float64"),
        "placement_distance": synapses["synapse_to_section_distance_um"].to_numpy(dtype="float64"),
        "source_vertex_distance": synapses[
            "synapse_to_cave_skeleton_vertex_distance_um"
        ].to_numpy(dtype="float64"),
    }


def _write_edges(
    path: Path,
    population_name: str,
    source_population: str,
    target_population: str,
    edge_type_id: int,
    arrays: Mapping[str, np.ndarray],
    source_count: int,
    target_count: int,
) -> None:
    """Write one edge per CAVE synapse and build bidirectional SONATA indices."""

    count = len(arrays["source_node_id"])
    if any(len(value) != count for value in arrays.values()):
        raise ValueError("SONATA edge property lengths disagree.")

    def writer(target: Path) -> None:
        with h5py.File(target, "w") as handle:
            population = handle.create_group(f"edges/{population_name}")
            source = population.create_dataset(
                "source_node_id", data=arrays["source_node_id"]
            )
            target_nodes = population.create_dataset(
                "target_node_id", data=arrays["target_node_id"]
            )
            source.attrs["node_population"] = source_population
            target_nodes.attrs["node_population"] = target_population
            population.create_dataset(
                "edge_type_id", data=np.full(count, edge_type_id, dtype="uint64")
            )
            population.create_dataset("edge_group_id", data=np.zeros(count, dtype="uint32"))
            population.create_dataset("edge_group_index", data=np.arange(count, dtype="uint64"))
            group = population.create_group("0")
            for name, values in arrays.items():
                if name in {"source_node_id", "target_node_id"}:
                    continue
                dataset = group.create_dataset(name, data=values)
                if name in {
                    "afferent_center_x", "afferent_center_y", "afferent_center_z",
                    "placement_distance", "source_vertex_distance",
                }:
                    dataset.attrs["units"] = "micrometre"

    _atomic_h5(path, writer)
    libsonata.EdgePopulation.write_indices(
        str(path), population_name, int(source_count), int(target_count), False
    )


def _copy_morphologies(
    morphology_manifest: pd.DataFrame,
    project_root: Path,
    destination: Path,
) -> list[str]:
    """Copy verified morphologies into the standalone SONATA package."""

    destination.mkdir(parents=True, exist_ok=True)
    names = []
    for row in morphology_manifest.sort_values("model_node_id").itertuples(index=False):
        source = project_root / str(row.simulation_morphology)
        target = destination / source.name
        if target.exists():
            if sha256_file(target) != sha256_file(source):
                raise RuntimeError(f"SONATA morphology copy differs: {target}")
        else:
            shutil.copy2(source, target)
        names.append(source.name)
    return names


def build_structural_sonata(
    final_manifest: pd.DataFrame,
    morphology_manifest: pd.DataFrame,
    intrinsic_synapses: pd.DataFrame,
    incoming_external_synapses: pd.DataFrame,
    project_root: str | Path,
    output_root: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the selected, virtual-external, and observed-edge SONATA package."""

    root = Path(project_root).resolve()
    output = Path(output_root).resolve()
    node_dir = output / "network/nodes"
    edge_dir = output / "network/edges"
    morphology_dir = output / "components/morphologies"

    population_names = get_sonata_population_names(config)
    selected_population = population_names["selected"]
    external_population = population_names["external"]
    intrinsic_edge_population = population_names["intrinsic_edges"]
    external_edge_population = population_names["external_edges"]

    morphology_names = _copy_morphologies(
        morphology_manifest,
        root,
        morphology_dir,
    )

    selected = final_manifest.sort_values("model_node_id").reset_index(drop=True)
    model_by_root = (
        selected
        .set_index("pt_root_id")["model_node_id"]
        .astype(int)
        .to_dict()
    )

    external_roots = sorted(
        incoming_external_synapses["pre_pt_root_id"].astype(int).unique()
    )
    external_manifest = pd.DataFrame(
        {
            "external_node_id": np.arange(
                len(external_roots),
                dtype="int64",
            ),
            "pt_root_id": np.asarray(
                external_roots,
                dtype="int64",
            ),
        }
    )
    external_by_root = (
        external_manifest
        .set_index("pt_root_id")["external_node_id"]
        .to_dict()
    )

    selected_nodes = node_dir / f"{selected_population}_nodes.h5"
    selected_types = node_dir / f"{selected_population}_node_types.csv"
    external_nodes = node_dir / f"{external_population}_nodes.h5"
    external_types = node_dir / f"{external_population}_node_types.csv"

    _write_selected_nodes(
        selected_nodes,
        selected,
        morphology_names,
        selected_population,
    )
    _write_external_nodes(
        external_nodes,
        external_manifest,
        external_population,
    )

    _atomic_csv(
        pd.DataFrame(
            [
                {
                    "node_type_id": 0,
                    "population": selected_population,
                    "model_type": "biophysical",
                    "recenter": 0,
                    "scientific_status": "awaiting_emodel_attachment",
                }
            ]
        ),
        selected_types,
    )
    _atomic_csv(
        pd.DataFrame(
            [
                {
                    "node_type_id": 100,
                    "population": external_population,
                    "model_type": "virtual",
                    "scientific_status": "observed_root_without_spike_train",
                }
            ]
        ),
        external_types,
    )

    intrinsic_ordered = intrinsic_synapses.sort_values("id").reset_index(drop=True)
    intrinsic_arrays = _edge_arrays(
        intrinsic_ordered,
        intrinsic_ordered["pre_pt_root_id"].map(model_by_root).to_numpy(),
        intrinsic_ordered["post_pt_root_id"].map(model_by_root).to_numpy(),
    )

    external_ordered = incoming_external_synapses.sort_values("id").reset_index(drop=True)
    external_arrays = _edge_arrays(
        external_ordered,
        external_ordered["pre_pt_root_id"].map(external_by_root).to_numpy(),
        external_ordered["post_pt_root_id"].map(model_by_root).to_numpy(),
    )

    intrinsic_edges = edge_dir / f"{intrinsic_edge_population}_edges.h5"
    intrinsic_types = edge_dir / f"{intrinsic_edge_population}_edge_types.csv"
    external_edges = edge_dir / f"{external_edge_population}_edges.h5"
    external_edge_types = edge_dir / f"{external_edge_population}_edge_types.csv"

    _write_edges(
        intrinsic_edges,
        intrinsic_edge_population,
        selected_population,
        selected_population,
        0,
        intrinsic_arrays,
        len(selected),
        len(selected),
    )
    _write_edges(
        external_edges,
        external_edge_population,
        external_population,
        selected_population,
        1,
        external_arrays,
        len(external_manifest),
        len(selected),
    )

    for frame, path in [
        (
            pd.DataFrame(
                [
                    {
                        "edge_type_id": 0,
                        "population": intrinsic_edge_population,
                        "scientific_status": "observed_structure_physiology_unassigned",
                    }
                ]
            ),
            intrinsic_types,
        ),
        (
            pd.DataFrame(
                [
                    {
                        "edge_type_id": 1,
                        "population": external_edge_population,
                        "scientific_status": "observed_structure_physiology_unassigned",
                    }
                ]
            ),
            external_edge_types,
        ),
    ]:
        _atomic_csv(frame, path)

    circuit = {
        "manifest": {
            "$BASE_DIR": ".",
            "$NETWORK_DIR": "$BASE_DIR/network",
            "$COMPONENT_DIR": "$BASE_DIR/components",
        },
        "components": {
            "morphologies_dir": "$COMPONENT_DIR/morphologies",
        },
        "networks": {
            "nodes": [
                {
                    "nodes_file": f"$NETWORK_DIR/nodes/{selected_nodes.name}",
                    "node_types_file": f"$NETWORK_DIR/nodes/{selected_types.name}",
                },
                {
                    "nodes_file": f"$NETWORK_DIR/nodes/{external_nodes.name}",
                    "node_types_file": f"$NETWORK_DIR/nodes/{external_types.name}",
                },
            ],
            "edges": [
                {
                    "edges_file": f"$NETWORK_DIR/edges/{intrinsic_edges.name}",
                    "edge_types_file": f"$NETWORK_DIR/edges/{intrinsic_types.name}",
                },
                {
                    "edges_file": f"$NETWORK_DIR/edges/{external_edges.name}",
                    "edge_types_file": f"$NETWORK_DIR/edges/{external_edge_types.name}",
                },
            ],
        },
    }

    circuit_path = output / "circuit_config.json"
    circuit_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(circuit, indent=2, sort_keys=True) + "\n"
    temporary = circuit_path.with_suffix(".json.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, circuit_path)

    return {
        "circuit_config": circuit_path,
        "selected_nodes": selected_nodes,
        "selected_node_types": selected_types,
        "external_nodes": external_nodes,
        "external_node_types": external_types,
        "intrinsic_edges": intrinsic_edges,
        "intrinsic_edge_types": intrinsic_types,
        "external_edges": external_edges,
        "external_edge_types": external_edge_types,
        "morphology_paths": [
            morphology_dir / name
            for name in morphology_names
        ],
        "external_manifest": external_manifest,
        "population_names": population_names,
    }


def validate_libsonata(build: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the structural SONATA package with BMTK and libSONATA."""

    from bmtk.utils.sonata.config import SonataConfig

    names = build["population_names"]
    selected_population = names["selected"]
    external_population = names["external"]
    intrinsic_edge_population = names["intrinsic_edges"]
    external_edge_population = names["external_edges"]

    selected_storage = libsonata.NodeStorage(str(build["selected_nodes"]))
    external_storage = libsonata.NodeStorage(str(build["external_nodes"]))
    selected = selected_storage.open_population(selected_population)
    external = external_storage.open_population(external_population)

    intrinsic_storage = libsonata.EdgeStorage(str(build["intrinsic_edges"]))
    external_edge_storage = libsonata.EdgeStorage(str(build["external_edges"]))
    intrinsic = intrinsic_storage.open_population(intrinsic_edge_population)
    external_edges = external_edge_storage.open_population(external_edge_population)

    circuit = SonataConfig.from_json(str(build["circuit_config"]))
    if not circuit.validate() or not circuit.with_networks:
        raise RuntimeError("BMTK rejected the structural SONATA circuit config.")

    selected_types = pd.read_csv(build["selected_node_types"])
    external_types = pd.read_csv(build["external_node_types"])
    intrinsic_types = pd.read_csv(build["intrinsic_edge_types"])
    external_edge_types = pd.read_csv(build["external_edge_types"])

    expected_type_populations = {
        "selected node types": (
            selected_types,
            selected_population,
        ),
        "external node types": (
            external_types,
            external_population,
        ),
        "intrinsic edge types": (
            intrinsic_types,
            intrinsic_edge_population,
        ),
        "external edge types": (
            external_edge_types,
            external_edge_population,
        ),
    }

    for label, (frame, expected_population) in expected_type_populations.items():
        if "population" not in frame.columns:
            raise RuntimeError(f"{label} lacks required population column.")
        observed = set(frame["population"].astype(str))
        if observed != {expected_population}:
            raise RuntimeError(
                f"{label} population mismatch: {observed} != "
                f"{{{expected_population!r}}}."
            )

    if "recenter" not in selected_types.columns:
        raise RuntimeError("Selected node types lack explicit recenter semantics.")
    if not selected_types["recenter"].astype(int).eq(0).all():
        raise RuntimeError(
            "Global-coordinate morphologies require recenter=0."
        )

    return {
        "libsonata_loadable": True,
        "bmtk_circuit_config_valid": True,
        "n_configured_node_files": int(len(circuit.nodes)),
        "n_configured_edge_files": int(len(circuit.edges)),
        "n_selected_nodes": int(len(selected)),
        "n_external_nodes": int(len(external)),
        "n_intrinsic_edges": int(len(intrinsic)),
        "n_external_edges": int(len(external_edges)),
        "selected_attributes": sorted(selected.attribute_names),
        "intrinsic_edge_attributes": sorted(intrinsic.attribute_names),
        "external_edge_attributes": sorted(external_edges.attribute_names),
        "selected_population": selected_population,
        "external_population": external_population,
        "intrinsic_edge_population": intrinsic_edge_population,
        "external_edge_population": external_edge_population,
        "recenter_disabled": True,
        "type_table_populations_valid": True,
    }


def run_stage09(project_root: str | Path | None = None) -> dict[str, Any]:
    """Build the structural SONATA package without assigning physiology."""

    from microns20.artifacts import write_dataframe
    from microns20.config import configured_path
    from microns20.orchestration import artifact_path, project_context
    from microns20.provenance import require_completed_stage, write_stage_provenance
    from microns20.validation import validate_frozen_manifest

    root, config = project_context(project_root)
    require_completed_stage("08_synapse_mapping", root, config)
    manifest_path = root / "data/processed/final20_manifest.parquet"
    mapping_path = root / "data/processed/final20_functional_mappings.parquet"
    morphology_manifest_path = configured_path(root, config, "processed_morphologies") / "manifest.parquet"
    connectivity_dir = configured_path(root, config, "processed_connectivity")
    intrinsic_path = connectivity_dir / "intrinsic_synapses.parquet"
    incoming_path = connectivity_dir / "incoming_external_synapses.parquet"
    manifest = pd.read_parquet(manifest_path)
    mappings = pd.read_parquet(mapping_path)
    morphology_manifest = pd.read_parquet(morphology_manifest_path)
    intrinsic = pd.read_parquet(intrinsic_path)
    incoming = pd.read_parquet(incoming_path)
    validate_frozen_manifest(manifest, mappings, config)
    if not manifest["model_node_id"].equals(morphology_manifest.sort_values("model_node_id")["model_node_id"].reset_index(drop=True)):
        raise RuntimeError("Morphology manifest does not preserve frozen model IDs.")
    output_root = configured_path(root, config, "processed_sonata")
    build = build_structural_sonata(
        manifest,
        morphology_manifest,
        intrinsic,
        incoming,
        root,
        output_root,
        config,
    )
    validation = validate_libsonata(build)
    if validation["n_selected_nodes"] != int(config["selection"]["n_neurons"]):
        raise RuntimeError("Structural SONATA selected-node count is incorrect.")
    if validation["n_intrinsic_edges"] != len(intrinsic) or validation["n_external_edges"] != len(incoming):
        raise RuntimeError("Structural SONATA lost or added synapses.")
    if len(build["morphology_paths"]) != int(config["selection"]["n_neurons"]):
        raise RuntimeError("Structural SONATA morphology count is incorrect.")
    external_manifest_path = connectivity_dir / "external_presynaptic_nodes.parquet"
    build_qc_path = artifact_path(root, config, "results_tables", "structural_sonata_build_qc.parquet")
    write_dataframe(build["external_manifest"], external_manifest_path, overwrite=True)
    build_qc = pd.DataFrame([{**validation, "n_sonata_morphologies": int(len(build["morphology_paths"])), "n_intrinsic_source_synapses": int(len(intrinsic)), "n_external_source_synapses": int(len(incoming)), "one_edge_per_source_synapse": True, "physiology_assigned": False, "emodels_attached": False, "simulation_config_emitted": False}])
    write_dataframe(build_qc, build_qc_path, overwrite=True)
    package_outputs = [build["circuit_config"], build["selected_nodes"], build["selected_node_types"], build["external_nodes"], build["external_node_types"], build["intrinsic_edges"], build["intrinsic_edge_types"], build["external_edges"], build["external_edge_types"], *build["morphology_paths"]]
    provenance = write_stage_provenance(
        "09_structural_sonata", root, config,
        inputs=[manifest_path, mapping_path, morphology_manifest_path, intrinsic_path, incoming_path],
        outputs=[*package_outputs, external_manifest_path, build_qc_path],
        source_metadata={
            "format": "SONATA",
            "selected_population": build["population_names"]["selected"],
            "external_population": build["population_names"]["external"],
            "physiology_status": "unassigned",
            "morphology_coordinate_policy": "global_coordinates_recenter_disabled",
        },
        summaries={**validation, "n_sonata_morphologies": int(len(build["morphology_paths"])), "one_edge_per_source_synapse": True, "structural_sonata_valid": True, "ready_for_emodel_attachment": True, "ready_for_biophysical_simulation": False},
    )
    return {"build": build, "build_qc": build_qc, "provenance": provenance}
