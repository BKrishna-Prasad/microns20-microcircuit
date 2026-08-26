"""Source-consistent CAVE synapse classification and morphology placement."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bmtk.builder.bionet import SWCReader
import numpy as np
import pandas as pd

from microns20.artifacts import require_columns
from microns20.cave import materialization_timestamp, validate_synapse_rows
from microns20.morphology import read_swc


TARGET_MAPPING_COLUMNS = [
    "post_pt_level2_id",
    "cave_skeleton_vertex_id",
    "raw_cave_point_id",
    "simulation_point_id",
    "afferent_section_id",
    "afferent_section_pos",
    "afferent_section_type",
    "afferent_center_x_um",
    "synapse_to_cave_skeleton_vertex_distance_um",
    "afferent_center_y_um",
    "afferent_center_z_um",
    "synapse_to_section_distance_um",
    "mapping_method",
    "mapping_status",
    "mapping_error",
]


def classify_synapses(
    incoming: pd.DataFrame,
    outgoing: pd.DataFrame,
    selected_roots: set[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split CAVE queries into intrinsic, positive incoming, outgoing, unresolved."""

    incoming = validate_synapse_rows(incoming)
    outgoing = validate_synapse_rows(outgoing)
    intrinsic = incoming.loc[
        incoming["pre_pt_root_id"].astype("int64").isin(selected_roots)
        & incoming["post_pt_root_id"].astype("int64").isin(selected_roots)
    ].copy()
    incoming_external = incoming.loc[
        incoming["pre_pt_root_id"].astype("int64").gt(0)
        & ~incoming["pre_pt_root_id"].astype("int64").isin(selected_roots)
        & incoming["post_pt_root_id"].astype("int64").isin(selected_roots)
    ].copy()
    zero_incoming = incoming.loc[
        incoming["pre_pt_root_id"].astype("int64").eq(0)
        & incoming["post_pt_root_id"].astype("int64").isin(selected_roots)
    ].copy()
    if not zero_incoming.empty:
        zero_incoming["mapping_status"] = "unresolved_zero_presynaptic_root"
        zero_incoming["mapping_error"] = (
            "CAVE reports pre_pt_root_id=0; no external source identity can be created."
        )
    outgoing_external = outgoing.loc[
        outgoing["pre_pt_root_id"].astype("int64").isin(selected_roots)
        & ~outgoing["post_pt_root_id"].astype("int64").isin(selected_roots)
    ].copy()

    incoming_ids = set(incoming["id"].astype("int64"))
    classified_ids = (
        set(intrinsic["id"].astype("int64"))
        | set(incoming_external["id"].astype("int64"))
        | set(zero_incoming["id"].astype("int64"))
    )
    if incoming_ids != classified_ids:
        missing = sorted(incoming_ids - classified_ids)
        raise ValueError(f"Incoming synapse classification lost IDs: {missing[:20]}")
    if len(intrinsic) + len(incoming_external) + len(zero_incoming) != len(incoming):
        raise ValueError("Incoming synapse classes overlap.")
    return intrinsic, incoming_external, outgoing_external, zero_incoming


def _batch_level2_roots(
    client: Any,
    supervoxels: np.ndarray,
    timestamp: Any,
    batch_size: int = 5_000,
) -> np.ndarray:
    """Resolve supervoxels to level-2 roots at the pinned materialization time."""

    values = np.asarray(supervoxels, dtype=np.uint64)
    outputs = []
    for start in range(0, len(values), int(batch_size)):
        outputs.append(
            np.asarray(
                client.chunkedgraph.get_roots(
                    values[start : start + batch_size],
                    timestamp=timestamp,
                    stop_layer=2,
                ),
                dtype=np.uint64,
            )
        )
    result = np.concatenate(outputs) if outputs else np.asarray([], dtype=np.uint64)
    if len(result) != len(values):
        raise RuntimeError("Level-2 root resolution changed synapse multiplicity.")
    return result


def _skeleton_level2_lookup(skeleton: Mapping[str, Any]) -> dict[int, list[int]]:
    """Map each level-2 ID to all skeleton vertices supported by its mesh vertices."""

    level2 = np.asarray(skeleton["lvl2_ids"], dtype=np.uint64)
    mesh_to_skeleton = np.asarray(skeleton["mesh_to_skel_map"], dtype=np.int64)
    if len(level2) != len(mesh_to_skeleton):
        raise ValueError("Skeleton level-2 and mesh mapping lengths disagree.")
    if np.any(mesh_to_skeleton < 0):
        raise ValueError("Skeleton mesh-to-skeleton map contains negative indices.")
    lookup: dict[int, set[int]] = defaultdict(set)
    for level2_id, vertex_id in zip(level2, mesh_to_skeleton):
        lookup[int(level2_id)].add(int(vertex_id))
    return {key: sorted(values) for key, values in lookup.items()}


def _verify_service_skeleton(
    skeleton: Mapping[str, Any],
    raw_swc: pd.DataFrame,
    expected_root_id: int,
) -> np.ndarray:
    """Return exact dict-vertex to raw-SWC IDs after integer-nm validation."""

    if int(skeleton["meta"]["root_id"]) != int(expected_root_id):
        raise ValueError("Skeleton service returned the wrong root identity.")
    vertices_nm = np.rint(
        np.asarray(skeleton["vertices"], dtype=float)
    ).astype("int64")
    compartments = np.asarray(skeleton["compartment"], dtype=np.int64)
    ordered = raw_swc.sort_values("id", kind="mergesort").reset_index(drop=True)
    if len(vertices_nm) != len(ordered):
        raise ValueError("Skeleton dict and raw SWC point counts disagree.")
    raw_nm = np.rint(
        ordered[["x", "y", "z"]].to_numpy(dtype=float) * 1_000.0
    ).astype("int64")
    service_keys = [
        (*xyz.tolist(), int(compartment))
        for xyz, compartment in zip(vertices_nm, compartments)
    ]
    raw_keys = [
        (*xyz.tolist(), int(compartment))
        for xyz, compartment in zip(
            raw_nm, ordered["type"].to_numpy(dtype="int64")
        )
    ]
    if len(set(service_keys)) != len(service_keys):
        raise ValueError("Skeleton dict has duplicate coordinate/type keys.")
    if len(set(raw_keys)) != len(raw_keys):
        raise ValueError("Raw SWC has duplicate coordinate/type keys.")
    raw_id_by_key = dict(zip(raw_keys, ordered["id"].astype(int)))
    if set(service_keys) != set(raw_keys):
        raise ValueError("Skeleton dict and raw SWC coordinate/type sets disagree.")
    return np.asarray(
        [raw_id_by_key[key] for key in service_keys], dtype=np.int64
    )


def _section_projection(
    reader: SWCReader,
    section_id: int,
    point_um: np.ndarray,
) -> tuple[float, np.ndarray, float]:
    """Project a point onto a concrete NEURON section polyline."""

    section = reader.get_section(int(section_id))
    coordinates = np.asarray(
        [
            [section.x3d(index), section.y3d(index), section.z3d(index)]
            for index in range(int(section.n3d()))
        ],
        dtype=float,
    )
    if len(coordinates) == 0:
        raise ValueError(f"NEURON section {section_id} has no 3-D points.")
    if len(coordinates) == 1:
        distance = float(np.linalg.norm(point_um - coordinates[0]))
        return 0.5, coordinates[0], distance
    vectors = coordinates[1:] - coordinates[:-1]
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(lengths <= 0):
        keep = lengths > 0
        vectors = vectors[keep]
        starts = coordinates[:-1][keep]
        lengths = lengths[keep]
    else:
        starts = coordinates[:-1]
    if not len(lengths):
        distance = float(np.linalg.norm(point_um - coordinates[0]))
        return 0.5, coordinates[0], distance
    fractions = np.clip(
        np.einsum("ij,ij->i", point_um - starts, vectors) / np.square(lengths),
        0.0,
        1.0,
    )
    projected = starts + fractions[:, None] * vectors
    distances = np.linalg.norm(projected - point_um, axis=1)
    best = int(np.argmin(distances))
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    along = float(cumulative[best] + fractions[best] * lengths[best])
    total = float(cumulative[-1])
    position = float(along / total)
    if not 0.0 <= position <= 1.0:
        raise AssertionError("Computed section position is outside [0, 1].")
    return position, projected[best], float(distances[best])


def map_postsynaptic_locations(
    client: Any,
    synapses: pd.DataFrame,
    final_manifest: pd.DataFrame,
    morphology_manifest: pd.DataFrame,
    point_mapping: pd.DataFrame,
    project_root: str | Path,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map target supervoxels through level 2 into exact simulation sections."""

    require_columns(
        synapses,
        {"id", "post_pt_root_id", "post_pt_supervoxel_id", "post_pt_position"},
        "postsynaptic CAVE synapses",
    )
    identity = final_manifest[["model_node_id", "nucleus_id", "pt_root_id"]]
    morphology_identity = morphology_manifest[[
        "model_node_id", "nucleus_id", "pt_root_id", "simulation_morphology"
    ]]
    targets = identity.merge(
        morphology_identity,
        on=["model_node_id", "nucleus_id", "pt_root_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(targets) != len(final_manifest):
        raise ValueError("Morphology manifest does not cover every frozen target.")
    target_by_root = targets.set_index("pt_root_id")
    timestamp = materialization_timestamp(client, config)
    mapped_tables: list[pd.DataFrame] = []
    unresolved_rows: list[dict[str, Any]] = []

    for root_id, group in synapses.groupby("post_pt_root_id", sort=True):
        root_id = int(root_id)
        if root_id not in target_by_root.index:
            raise ValueError(f"Synapse target root is not frozen: {root_id}")
        target = target_by_root.loc[root_id]
        raw_path = Path(project_root) / str(
            final_manifest.loc[
                final_manifest["pt_root_id"].eq(root_id), "cave_skeleton_file"
            ].iloc[0]
        )
        simulation_path = Path(project_root) / str(target["simulation_morphology"])
        raw_swc = read_swc(raw_path)
        simulation_swc = read_swc(simulation_path)
        skeleton = client.skeleton.get_skeleton(
            root_id,
            output_format="dict",
            skeleton_version=int(config["cave"]["skeleton_version"]),
        )
        vertex_to_raw = _verify_service_skeleton(skeleton, raw_swc, root_id)
        level2_lookup = _skeleton_level2_lookup(skeleton)
        vertices_um = np.asarray(skeleton["vertices"], dtype=float) / 1_000.0
        level2_ids = _batch_level2_roots(
            client,
            group["post_pt_supervoxel_id"].astype("uint64").to_numpy(),
            timestamp,
        )
        reader = SWCReader(str(simulation_path))
        section_by_key = {
            (int(reader._get_sec_type(section)), int(reader._get_sec_nameindex(section))): section_id
            for section_id, section in enumerate(reader.sections)
        }
        if len(section_by_key) != len(reader.sections):
            raise ValueError("BMTK sections do not have unique type/nameindex keys.")
        swc_section_key = reader.swc_map.set_index("id")[["type", "nameindex"]]
        raw_to_simulation = (
            point_mapping.loc[point_mapping["pt_root_id"].eq(root_id)]
            .set_index("raw_point_id")["simulation_point_id"]
            .astype("int64")
            .to_dict()
        )
        if len(raw_to_simulation) != len(raw_swc):
            raise ValueError("Point normalization map does not cover the target skeleton.")
        records = []
        for source_row, level2_id in zip(group.itertuples(index=False), level2_ids):
            record = source_row._asdict()
            try:
                candidates = level2_lookup.get(int(level2_id), [])
                if not candidates:
                    raise KeyError(f"Level-2 ID {int(level2_id)} is absent from skeleton map.")
                point_um = np.asarray(source_row.post_pt_position, dtype=float)
                if point_um.shape != (3,) or not np.isfinite(point_um).all():
                    raise ValueError("Post-synaptic coordinate is not a finite 3-vector.")
                candidate_coordinates = vertices_um[np.asarray(candidates, dtype=int)]
                candidate_distances = np.linalg.norm(
                    candidate_coordinates - point_um, axis=1
                )
                best_candidate = int(np.argmin(candidate_distances))
                vertex_id = int(candidates[best_candidate])
                vertex_distance = float(candidate_distances[best_candidate])
                raw_point_id = int(vertex_to_raw[vertex_id])
                simulation_point_id = int(raw_to_simulation[raw_point_id])
                section_row = swc_section_key.loc[simulation_point_id]
                section_id = int(
                    section_by_key[
                        (int(section_row["type"]), int(section_row["nameindex"]))
                    ]
                )
                section_pos, center, distance = _section_projection(
                    reader, section_id, point_um
                )
                section_type = int(
                    simulation_swc.loc[
                        simulation_swc["id"].eq(simulation_point_id), "type"
                    ].iloc[0]
                )
                record.update(
                    {
                        "model_node_id": int(target["model_node_id"]),
                        "target_nucleus_id": int(target["nucleus_id"]),
                        "post_pt_level2_id": int(level2_id),
                        "cave_skeleton_vertex_id": vertex_id,
                        "raw_cave_point_id": raw_point_id,
                        "simulation_point_id": simulation_point_id,
                        "afferent_section_id": section_id,
                        "afferent_section_pos": section_pos,
                        "afferent_section_type": section_type,
                        "afferent_center_x_um": float(center[0]),
                        "afferent_center_y_um": float(center[1]),
                        "afferent_center_z_um": float(center[2]),
                        "synapse_to_section_distance_um": distance,
                        "synapse_to_cave_skeleton_vertex_distance_um": vertex_distance,
                        "mapping_method": (
                            "post_supervoxel_to_pinned_level2_to_cave_skeleton_v4_"
                            "to_normalized_swc_to_neuron_section_projection"
                        ),
                        "mapping_status": "mapped",
                        "mapping_error": "",
                    }
                )
                records.append(record)
            except Exception as error:
                record.update(
                    {
                        "model_node_id": int(target["model_node_id"]),
                        "target_nucleus_id": int(target["nucleus_id"]),
                        "post_pt_level2_id": int(level2_id),
                        "mapping_status": "unresolved",
                        "mapping_error": f"{type(error).__name__}: {error}",
                    }
                )
                unresolved_rows.append(record)
        if records:
            mapped_tables.append(pd.DataFrame(records))
    mapped = pd.concat(mapped_tables, ignore_index=True) if mapped_tables else pd.DataFrame()
    unresolved = pd.DataFrame(unresolved_rows)
    if len(mapped) + len(unresolved) != len(synapses):
        raise AssertionError("Synapse target mapping changed row multiplicity.")
    if not mapped.empty and mapped["id"].duplicated().any():
        raise ValueError("Mapped target synapse IDs are duplicated.")
    return mapped.sort_values("id").reset_index(drop=True), unresolved


def synapse_mapping_qc(
    intrinsic: pd.DataFrame,
    incoming_external: pd.DataFrame,
    unresolved: pd.DataFrame,
    final_manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize mapped target counts and descriptive placement distances."""

    mapped = pd.concat([intrinsic, incoming_external], ignore_index=True)
    rows = []
    for row in final_manifest.sort_values("model_node_id").itertuples(index=False):
        target = mapped.loc[mapped["model_node_id"].eq(int(row.model_node_id))]
        target_unresolved = (
            unresolved.loc[unresolved["post_pt_root_id"].eq(int(row.pt_root_id))]
            if not unresolved.empty and "post_pt_root_id" in unresolved
            else pd.DataFrame()
        )
        distances = target["synapse_to_section_distance_um"].to_numpy(dtype=float)
        rows.append(
            {
                "model_node_id": int(row.model_node_id),
                "nucleus_id": int(row.nucleus_id),
                "pt_root_id": int(row.pt_root_id),
                "n_mapped_intrinsic_synapses": int(
                    intrinsic["model_node_id"].eq(int(row.model_node_id)).sum()
                ),
                "n_mapped_external_incoming_synapses": int(
                    incoming_external["model_node_id"].eq(int(row.model_node_id)).sum()
                ),
                "n_unresolved_incoming_synapses": int(len(target_unresolved)),
                "placement_distance_median_um": (
                    float(np.median(distances)) if len(distances) else np.nan
                ),
                "placement_distance_p95_um": (
                    float(np.percentile(distances, 95)) if len(distances) else np.nan
                ),
                "placement_distance_max_um": (
                    float(np.max(distances)) if len(distances) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)



def run_stage08(project_root: str | Path | None = None) -> dict[str, Any]:
    """Query, classify, and source-consistently map CAVE synapses."""

    from microns20 import cave
    from microns20.artifacts import write_dataframe
    from microns20.config import configured_path
    from microns20.orchestration import artifact_path, project_context
    from microns20.provenance import require_completed_stage, write_stage_provenance

    root, config = project_context(project_root)
    require_completed_stage("07_simulation_morphologies", root, config)
    manifest_path = root / "data/processed/final20_manifest.parquet"
    morphology_manifest_path = configured_path(root, config, "processed_morphologies") / "manifest.parquet"
    point_mapping_path = artifact_path(root, config, "morphologies", "cave_to_simulation_point_map.parquet")
    manifest = pd.read_parquet(manifest_path)
    morphology_manifest = pd.read_parquet(morphology_manifest_path)
    point_mapping = pd.read_parquet(point_mapping_path)
    selected_roots = set(manifest["pt_root_id"].astype(int))
    client = cave.create_client(config)
    incoming = cave.query_synapses(client, config, post_ids=selected_roots, include_zeros=True)
    outgoing = cave.query_synapses(client, config, pre_ids=selected_roots, include_zeros=True)
    intrinsic_raw, incoming_external_raw, outgoing_external, zero_incoming = classify_synapses(incoming, outgoing, selected_roots)
    positive_targets = pd.concat([intrinsic_raw, incoming_external_raw], ignore_index=True).sort_values("id").reset_index(drop=True)
    mapped, mapping_failures = map_postsynaptic_locations(client, positive_targets, manifest, morphology_manifest, point_mapping, root, config)
    intrinsic = mapped.loc[mapped["pre_pt_root_id"].astype("int64").isin(selected_roots)].copy()
    incoming_external = mapped.loc[~mapped["pre_pt_root_id"].astype("int64").isin(selected_roots)].copy()
    if set(intrinsic["id"].astype(int)) != set(intrinsic_raw["id"].astype(int)):
        raise RuntimeError("Intrinsic mapping changed authoritative synapse IDs.")
    if set(incoming_external["id"].astype(int)) != set(incoming_external_raw["id"].astype(int)):
        raise RuntimeError("External mapping changed authoritative synapse IDs.")
    unresolved = pd.concat([zero_incoming, mapping_failures], ignore_index=True, sort=False)
    qc = synapse_mapping_qc(intrinsic, incoming_external, unresolved, manifest)
    output_dir = configured_path(root, config, "processed_connectivity")
    intrinsic_output = output_dir / "intrinsic_synapses.parquet"
    incoming_output = output_dir / "incoming_external_synapses.parquet"
    outgoing_output = output_dir / "outgoing_external_synapses.parquet"
    unresolved_output = output_dir / "unresolved_synapses.parquet"
    qc_output = artifact_path(root, config, "results_tables", "synapse_mapping_qc.parquet")
    for dataframe, path in [(intrinsic, intrinsic_output), (incoming_external, incoming_output), (outgoing_external, outgoing_output), (unresolved, unresolved_output), (qc, qc_output)]:
        write_dataframe(dataframe, path, overwrite=True)
    summaries = {
        "n_all_incoming_synapses": int(len(incoming)),
        "n_intrinsic_synapses": int(len(intrinsic)),
        "n_positive_external_incoming_synapses": int(len(incoming_external)),
        "n_zero_root_incoming_synapses": int(len(zero_incoming)),
        "n_positive_target_mapping_failures": int(len(mapping_failures)),
        "n_outgoing_external_synapses": int(len(outgoing_external)),
        "n_external_presynaptic_roots": int(incoming_external["pre_pt_root_id"].nunique()),
        "mapping_method": "supervoxel -> pinned level-2 -> skeleton-v4 vertex -> normalized SWC point -> BMTK/NEURON section projection",
        "placement_distance_um": {
            "median": float(mapped["synapse_to_section_distance_um"].median()),
            "p95": float(mapped["synapse_to_section_distance_um"].quantile(0.95)),
            "maximum": float(mapped["synapse_to_section_distance_um"].max()),
        },
    }
    status = "complete" if mapping_failures.empty else "blocked"
    provenance = write_stage_provenance(
        "08_synapse_mapping", root, config,
        inputs=[manifest_path, morphology_manifest_path, point_mapping_path],
        outputs=[intrinsic_output, incoming_output, outgoing_output, unresolved_output, qc_output],
        source_metadata={**cave.source_metadata(client, config), "coordinate_space": "MICrONS CAVE", "coordinate_unit": "micrometre"},
        summaries=summaries, status=status,
    )
    if not mapping_failures.empty:
        raise RuntimeError(f"{len(mapping_failures)} positive-root synapses could not be mapped; Stage 09 is forbidden.")
    return {"intrinsic_synapses": intrinsic, "incoming_external_synapses": incoming_external, "outgoing_external_synapses": outgoing_external, "unresolved_synapses": unresolved, "mapping_qc": qc, "provenance": provenance}
