"""CAVE skeleton retrieval, eligibility, and lossless SWC normalization."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bmtk.builder.bionet import SWCReader
import morphio
import networkx as nx
import numpy as np
import pandas as pd

from microns20.artifacts import (
    dataframe_digest,
    require_columns,
    sha256_file,
    write_text_once,
)
from microns20.config import cave_raw_directory, configured_path


SWC_COLUMNS = ["id", "type", "x", "y", "z", "radius", "parent"]
ALLOWED_SWC_TYPES = {1, 2, 3, 4}
SWC_TYPE_NAMES = {1: "soma", 2: "axon", 3: "dendrite", 4: "apical"}


def read_swc(path: str | Path) -> pd.DataFrame:
    """Read an SWC file without changing identifiers, units, or geometry."""

    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(target)
    dataframe = pd.read_csv(
        target,
        sep=r"\s+",
        names=SWC_COLUMNS,
        float_precision="round_trip",
        comment="#",
    )
    if dataframe.empty:
        raise ValueError(f"SWC is empty: {target}")
    for column in ("id", "type", "parent"):
        dataframe[column] = pd.to_numeric(
            dataframe[column], errors="raise"
        ).astype("int64")
    for column in ("x", "y", "z", "radius"):
        dataframe[column] = pd.to_numeric(
            dataframe[column], errors="raise"
        ).astype("float64")
    return dataframe


def swc_text(dataframe: pd.DataFrame) -> str:
    """Serialize SWC rows deterministically with full numeric precision."""

    require_columns(dataframe, SWC_COLUMNS, "SWC dataframe")
    lines = []
    for row in dataframe[SWC_COLUMNS].itertuples(index=False):
        lines.append(
            " ".join(
                [
                    str(int(row.id)),
                    str(int(row.type)),
                    format(float(row.x), ".17g"),
                    format(float(row.y), ".17g"),
                    format(float(row.z), ".17g"),
                    format(float(row.radius), ".17g"),
                    str(int(row.parent)),
                ]
            )
        )
    return "\n".join(lines) + "\n"


def raw_skeleton_directory(
    project_root: str | Path,
    config: Mapping[str, Any],
) -> Path:
    """Return the canonical versioned skeleton-v4 directory."""

    cave = config["cave"]
    return (
        cave_raw_directory(project_root, config)
        / "skeletons"
        / f"skeleton_v{int(cave['skeleton_version'])}"
    )


def raw_skeleton_path(
    project_root: str | Path,
    config: Mapping[str, Any],
    root_id: int,
) -> Path:
    """Return one stable-root raw skeleton path."""

    if int(root_id) <= 0:
        raise ValueError("CAVE root IDs must be positive.")
    return raw_skeleton_directory(project_root, config) / f"root_{int(root_id)}.swc"


def _service_swc(client: Any, root_id: int, skeleton_version: int) -> pd.DataFrame:
    downloaded = client.skeleton.get_skeleton(
        int(root_id),
        output_format="swc",
        skeleton_version=int(skeleton_version),
    )
    if not isinstance(downloaded, pd.DataFrame):
        raise TypeError(
            f"CAVE skeleton service returned {type(downloaded).__name__} "
            f"for root {root_id}."
        )
    require_columns(downloaded, SWC_COLUMNS, f"CAVE skeleton {root_id}")
    normalized = downloaded[SWC_COLUMNS].copy()
    for column in ("id", "type", "parent"):
        normalized[column] = normalized[column].astype("int64")
    for column in ("x", "y", "z", "radius"):
        normalized[column] = normalized[column].astype("float64")
    return normalized


def cache_skeleton(
    client: Any | None,
    project_root: str | Path,
    config: Mapping[str, Any],
    root_id: int,
) -> tuple[Path, str]:
    """Fetch one official skeleton or validate its immutable raw cache."""

    target = raw_skeleton_path(project_root, config, root_id)
    if client is None:
        if not target.is_file():
            raise FileNotFoundError(
                f"Immutable raw skeleton cache is absent for root {root_id}: {target}"
            )
        read_swc(target)
        return target, "immutable_raw_cache"
    service = _service_swc(
        client,
        int(root_id),
        int(config["cave"]["skeleton_version"]),
    )
    if target.exists():
        cached = read_swc(target)
        if dataframe_digest(cached) != dataframe_digest(service):
            raise RuntimeError(
                f"Cached raw skeleton differs from CAVE skeleton v"
                f"{config['cave']['skeleton_version']} for root {root_id}."
            )
        source = "validated_cache"
    else:
        write_text_once(swc_text(service), target)
        source = "cave_skeleton_service"
    return target, source


def validate_swc_tree(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Describe SWC geometry, topology, compartments, and radii."""

    require_columns(dataframe, SWC_COLUMNS, "SWC")
    ids = dataframe["id"].astype("int64")
    parents = dataframe["parent"].astype("int64")
    types = dataframe["type"].astype("int64")
    geometry = dataframe[["x", "y", "z"]].to_numpy(dtype=float)
    radii = dataframe["radius"].to_numpy(dtype=float)

    unique_ids = not ids.duplicated().any()
    root_rows = dataframe.loc[parents.eq(-1)]
    id_set = set(ids.astype(int))
    missing_parents = sorted(
        {
            int(parent)
            for parent in parents
            if int(parent) != -1 and int(parent) not in id_set
        }
    )
    self_parents = dataframe.loc[
        dataframe["id"].eq(dataframe["parent"]), "id"
    ].astype(int).tolist()

    graph = nx.DiGraph()
    graph.add_nodes_from(id_set)
    graph.add_edges_from(
        (int(parent), int(point))
        for point, parent in zip(ids, parents)
        if int(parent) != -1 and int(parent) in id_set
    )
    acyclic = nx.is_directed_acyclic_graph(graph)
    connected = (
        nx.is_weakly_connected(graph) if len(graph) > 0 else False
    )

    counts = types.value_counts().to_dict()
    n_type3 = int(counts.get(3, 0))
    n_apical = int(counts.get(4, 0))
    allowed = set(types.astype(int)).issubset(ALLOWED_SWC_TYPES)
    finite_coordinates = bool(np.isfinite(geometry).all())
    finite_radii = bool(np.isfinite(radii).all())
    positive_radii = bool(finite_radii and np.all(radii > 0))

    return {
        "n_skeleton_points": int(len(dataframe)),
        "n_roots": int(len(root_rows)),
        "root_point_id": (
            int(root_rows.iloc[0]["id"]) if len(root_rows) == 1 else None
        ),
        "root_type": (
            int(root_rows.iloc[0]["type"]) if len(root_rows) == 1 else None
        ),
        "n_soma_points": int(counts.get(1, 0)),
        "n_axon_points": int(counts.get(2, 0)),
        "n_basal_or_generic_dendrite_points": n_type3,
        "n_apical_points": n_apical,
        "n_dendrite_points": n_type3 + n_apical,
        "apical_annotation_present": bool(n_apical > 0),
        "n_unknown_points": int((~types.isin(ALLOWED_SWC_TYPES)).sum()),
        "finite_coordinates": finite_coordinates,
        "finite_radii": finite_radii,
        "positive_radii": positive_radii,
        "has_nonpositive_radius": bool(
            np.any(np.isfinite(radii) & (radii <= 0))
        ),
        "n_nonpositive_radius_points": int(
            np.sum(np.isfinite(radii) & (radii <= 0))
        ),
        "requires_radius_normalization": bool(finite_radii and np.any(radii <= 0)),
        "n_nonfinite_radii": int(np.sum(~np.isfinite(radii))),
        "unique_point_ids": bool(unique_ids),
        "valid_parent_references": bool(not missing_parents),
        "missing_parent_ids": missing_parents,
        "self_parent_point_ids": self_parents,
        "acyclic": bool(acyclic),
        "connected": bool(connected),
        "allowed_types": bool(allowed),
    }


def _compatibility(path: Path) -> dict[str, Any]:
    morphio_ok = False
    morphio_error = ""
    bmtk_ok = False
    bmtk_error = ""
    try:
        morphology = morphio.Morphology(str(path))
        _ = list(morphology.sections)
        morphio_ok = True
    except Exception as error:
        morphio_error = f"{type(error).__name__}: {error}"
    try:
        reader = SWCReader(str(path))
        _ = reader.sections
        _ = reader.soma_position
        bmtk_ok = True
    except Exception as error:
        bmtk_error = f"{type(error).__name__}: {error}"
    return {
        "morphio_loadable": morphio_ok,
        "morphio_error": morphio_error,
        "bmtk_neuron_loadable": bmtk_ok,
        "bmtk_neuron_error": bmtk_error,
    }


def evaluate_candidate_skeletons(
    client: Any | None,
    candidates: pd.DataFrame,
    project_root: str | Path,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch and independently validate every unique biological candidate."""

    require_columns(candidates, {"nucleus_id", "pt_root_id"}, "candidates")
    identities = candidates[["nucleus_id", "pt_root_id"]].drop_duplicates()
    if identities["nucleus_id"].duplicated().any():
        raise ValueError("A candidate nucleus maps to multiple CAVE roots.")
    if identities["pt_root_id"].duplicated().any():
        raise ValueError("A CAVE root maps to multiple candidate nuclei.")

    records = []
    for row in identities.sort_values("pt_root_id").itertuples(index=False):
        record: dict[str, Any] = {
            "nucleus_id": int(row.nucleus_id),
            "pt_root_id": int(row.pt_root_id),
            "skeleton_retrieved": False,
            "skeleton_error": "",
        }
        try:
            path, source = cache_skeleton(
                client,
                project_root,
                config,
                int(row.pt_root_id),
            )
            skeleton = read_swc(path)
            record.update(
                {
                    "cave_skeleton_file": str(
                        path.resolve().relative_to(Path(project_root).resolve())
                    ),
                    "cave_skeleton_source": source,
                    "cave_skeleton_sha256": sha256_file(path),
                    "skeleton_retrieved": True,
                    **validate_swc_tree(skeleton),
                    **_compatibility(path),
                }
            )
        except Exception as error:
            record["skeleton_error"] = f"{type(error).__name__}: {error}"
        records.append(record)

    qc = pd.DataFrame(records)
    rules = config["eligibility"]["skeleton"]
    required_checks = [
        qc["skeleton_retrieved"].fillna(False),
        qc["n_roots"].fillna(0).eq(int(rules["required_roots"])),
        qc["n_soma_points"].fillna(0).gt(0),
        qc["n_axon_points"].fillna(0).gt(0),
        qc["n_dendrite_points"].fillna(0).gt(0),
        qc["finite_coordinates"].fillna(False),
        qc["finite_radii"].fillna(False),
        qc["unique_point_ids"].fillna(False),
        qc["valid_parent_references"].fillna(False),
        qc["acyclic"].fillna(False),
        qc["connected"].fillna(False),
        qc["allowed_types"].fillna(False),
        qc["morphio_loadable"].fillna(False),
        qc["bmtk_neuron_loadable"].fillna(False),
    ]
    if bool(rules["require_apical"]):
        required_checks.append(qc["n_apical_points"].fillna(0).gt(0))
    eligible_mask = required_checks[0].copy()
    for check in required_checks[1:]:
        eligible_mask &= check
    qc["cave_morphology_eligible"] = eligible_mask

    merged = candidates.merge(
        qc,
        on=["nucleus_id", "pt_root_id"],
        how="left",
        validate="many_to_one",
    )
    eligible = merged.loc[merged["cave_morphology_eligible"]].copy()
    failures = merged.loc[~merged["cave_morphology_eligible"]].copy()
    return (
        eligible.sort_values(
            ["session", "scan_idx", "nucleus_id"]
        ).reset_index(drop=True),
        qc.sort_values("nucleus_id").reset_index(drop=True),
        failures.sort_values(
            ["session", "scan_idx", "nucleus_id"]
        ).reset_index(drop=True),
    )


def normalize_swc(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Renumber one valid tree parent-first while preserving all morphology."""

    qc = validate_swc_tree(dataframe)
    required = [
        qc["n_roots"] == 1,
        qc["unique_point_ids"],
        qc["valid_parent_references"],
        qc["acyclic"],
        qc["connected"],
        qc["finite_coordinates"],
        qc["positive_radii"],
        qc["allowed_types"],
    ]
    if not all(required):
        raise ValueError(f"Cannot normalize invalid SWC: {qc}")

    by_id = dataframe.set_index("id", drop=False)
    children: dict[int, list[int]] = {int(value): [] for value in by_id.index}
    root_id = int(dataframe.loc[dataframe["parent"].eq(-1), "id"].iloc[0])
    for row in dataframe.itertuples(index=False):
        if int(row.parent) != -1:
            children[int(row.parent)].append(int(row.id))
    for values in children.values():
        values.sort()

    order = []
    stack = [root_id]
    while stack:
        current = stack.pop()
        order.append(current)
        stack.extend(reversed(children[current]))
    if len(order) != len(dataframe):
        raise AssertionError("Topological traversal lost SWC points.")

    new_by_old = {old: index + 1 for index, old in enumerate(order)}
    rows = []
    mapping_rows = []
    for old in order:
        source = by_id.loc[old]
        parent = int(source["parent"])
        new_id = new_by_old[old]
        new_parent = -1 if parent == -1 else new_by_old[parent]
        rows.append(
            {
                "id": new_id,
                "type": int(source["type"]),
                "x": float(source["x"]),
                "y": float(source["y"]),
                "z": float(source["z"]),
                "radius": float(source["radius"]),
                "parent": new_parent,
            }
        )
        mapping_rows.append(
            {
                "raw_point_id": int(old),
                "simulation_point_id": int(new_id),
            }
        )
    return pd.DataFrame(rows, columns=SWC_COLUMNS), pd.DataFrame(mapping_rows)


def compartment_transition_events(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Record every parent-child compartment transition descriptively."""

    by_id = dataframe.set_index("id")
    rows = []
    for row in dataframe.itertuples(index=False):
        if int(row.parent) == -1:
            continue
        parent_type = int(by_id.loc[int(row.parent), "type"])
        child_type = int(row.type)
        if parent_type != child_type:
            rows.append(
                {
                    "parent_point_id": int(row.parent),
                    "child_point_id": int(row.id),
                    "parent_type": parent_type,
                    "child_type": child_type,
                    "parent_compartment": SWC_TYPE_NAMES.get(parent_type),
                    "child_compartment": SWC_TYPE_NAMES.get(child_type),
                }
            )
    return pd.DataFrame(rows)


def validate_simulation_morphology(path: str | Path) -> dict[str, Any]:
    """Validate one normalized SWC under MorphIO and BMTK/NEURON."""

    target = Path(path)
    dataframe = read_swc(target)
    tree = validate_swc_tree(dataframe)
    compatibility = _compatibility(target)
    return {
        **tree,
        **compatibility,
        "root_is_soma": bool(tree["root_type"] == 1),
        "simulation_morphology_valid": bool(
            tree["n_roots"] == 1
            and tree["n_soma_points"] > 0
            and tree["n_axon_points"] > 0
            and tree["n_dendrite_points"] > 0
            and tree["root_type"] == 1
            and tree["finite_coordinates"]
            and tree["positive_radii"]
            and tree["unique_point_ids"]
            and tree["valid_parent_references"]
            and tree["acyclic"]
            and tree["connected"]
            and tree["allowed_types"]
            and compatibility["morphio_loadable"]
            and compatibility["bmtk_neuron_loadable"]
        ),
    }


def create_simulation_morphologies(
    final_manifest: pd.DataFrame,
    project_root: str | Path,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create one deterministic, geometry-preserving SWC per frozen neuron."""

    require_columns(
        final_manifest,
        {"model_node_id", "nucleus_id", "pt_root_id", "cave_skeleton_file"},
        "final manifest",
    )
    output_dir = configured_path(
        project_root, config, "processed_morphologies"
    )
    manifest_rows = []
    qc_rows = []
    mapping_tables = []
    event_tables = []

    for row in final_manifest.sort_values("model_node_id").itertuples(index=False):
        raw_path = Path(project_root) / str(row.cave_skeleton_file)
        raw = read_swc(raw_path)
        normalized, point_mapping = normalize_swc(raw)
        filename = (
            f"model_{int(row.model_node_id):03d}_"
            f"nucleus_{int(row.nucleus_id)}_"
            f"root_{int(row.pt_root_id)}.swc"
        )
        target = output_dir / filename
        write_text_once(swc_text(normalized), target)

        reread = read_swc(target)
        if not np.array_equal(
            raw[["x", "y", "z", "radius", "type"]]
            .sort_index()
            .to_numpy(),
            reread.set_index("id")
            .loc[
                point_mapping.sort_values("simulation_point_id")[
                    "simulation_point_id"
                ],
                ["x", "y", "z", "radius", "type"],
            ]
            .to_numpy(),
        ):
            raw_by_order = raw.set_index("id").loc[
                point_mapping.sort_values("simulation_point_id")["raw_point_id"],
                ["x", "y", "z", "radius", "type"],
            ]
            new_by_order = reread.sort_values("id")[
                ["x", "y", "z", "radius", "type"]
            ]
            if not np.array_equal(
                raw_by_order.to_numpy(), new_by_order.to_numpy()
            ):
                raise ValueError(
                    f"Geometry changed during normalization for {target}."
                )

        morphology_qc = validate_simulation_morphology(target)
        if not morphology_qc["simulation_morphology_valid"]:
            raise ValueError(
                f"Normalized simulation morphology failed QC: "
                f"{target}: {morphology_qc}"
            )
        manifest_rows.append(
            {
                "model_node_id": int(row.model_node_id),
                "nucleus_id": int(row.nucleus_id),
                "pt_root_id": int(row.pt_root_id),
                "source_cave_skeleton": str(row.cave_skeleton_file),
                "simulation_morphology": str(
                    target.resolve().relative_to(Path(project_root).resolve())
                ),
                "sha256": sha256_file(target),
            }
        )
        qc_rows.append(
            {
                "model_node_id": int(row.model_node_id),
                "nucleus_id": int(row.nucleus_id),
                "pt_root_id": int(row.pt_root_id),
                "simulation_morphology": str(
                    target.resolve().relative_to(Path(project_root).resolve())
                ),
                **morphology_qc,
            }
        )
        point_mapping.insert(0, "pt_root_id", int(row.pt_root_id))
        point_mapping.insert(0, "model_node_id", int(row.model_node_id))
        mapping_tables.append(point_mapping)
        events = compartment_transition_events(normalized)
        if not events.empty:
            events.insert(0, "pt_root_id", int(row.pt_root_id))
            events.insert(0, "model_node_id", int(row.model_node_id))
            event_tables.append(events)

    point_map = pd.concat(mapping_tables, ignore_index=True)
    events = (
        pd.concat(event_tables, ignore_index=True)
        if event_tables
        else pd.DataFrame(
            columns=[
                "model_node_id",
                "pt_root_id",
                "parent_point_id",
                "child_point_id",
                "parent_type",
                "child_type",

                "parent_compartment",
                "child_compartment",
            ]
        )
    )
    return (
        pd.DataFrame(manifest_rows),
        pd.DataFrame(qc_rows),
        point_map,
        events,
    )
def normalize_nonpositive_radii(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace finite nonpositive radii by graph-harmonic interpolation.

    Positive source radii are fixed boundary values.  For each connected set of
    nonpositive-radius points, the corrected radii solve the discrete Laplace
    equation on the SWC tree.  This is one cell-independent, topology-aware,
    data-derived rule and introduces no fixed radius constant.
    """

    require_columns(dataframe, SWC_COLUMNS, "raw CAVE SWC")
    radii = dataframe["radius"].to_numpy(dtype=float)
    if not np.isfinite(radii).all():
        raise ValueError("Non-finite source radii cannot be normalized.")
    corrected = dataframe.copy()
    bad_ids = set(
        corrected.loc[corrected["radius"].le(0), "id"].astype(int).tolist()
    )
    columns = [
        "raw_point_id", "raw_radius_um", "normalized_radius_um",
        "nonpositive_component_size", "positive_boundary_point_ids",
        "positive_boundary_radii_um", "normalization_rule",
    ]
    if not bad_ids:
        return corrected, pd.DataFrame(columns=columns)

    graph = nx.Graph()
    graph.add_nodes_from(corrected["id"].astype(int))
    graph.add_edges_from(
        (int(row.parent), int(row.id))
        for row in corrected.itertuples(index=False)
        if int(row.parent) != -1
    )
    radius_by_id = corrected.set_index("id")["radius"].astype(float).to_dict()
    events: list[dict[str, Any]] = []
    for component in nx.connected_components(graph.subgraph(bad_ids)):
        unknown = sorted(int(value) for value in component)
        unknown_set = set(unknown)
        boundary = sorted(
            {
                int(neighbor)
                for point_id in unknown
                for neighbor in graph.neighbors(point_id)
                if int(neighbor) not in unknown_set
                and float(radius_by_id[int(neighbor)]) > 0
            }
        )
        if not boundary:
            raise ValueError(
                "A nonpositive-radius component has no positive topological boundary: "
                f"{unknown}"
            )
        index = {point_id: offset for offset, point_id in enumerate(unknown)}
        matrix = np.zeros((len(unknown), len(unknown)), dtype=float)
        right = np.zeros(len(unknown), dtype=float)
        for point_id in unknown:
            row_index = index[point_id]
            neighbors = [int(value) for value in graph.neighbors(point_id)]
            matrix[row_index, row_index] = float(len(neighbors))
            for neighbor in neighbors:
                if neighbor in unknown_set:
                    matrix[row_index, index[neighbor]] -= 1.0
                else:
                    neighbor_radius = float(radius_by_id[neighbor])
                    if neighbor_radius <= 0:
                        raise AssertionError("Radius component boundary is not positive.")
                    right[row_index] += neighbor_radius
        solved = np.linalg.solve(matrix, right)
        if not np.isfinite(solved).all() or np.any(solved <= 0):
            raise ValueError(
                f"Topology-harmonic radius normalization failed for {unknown}."
            )
        boundary_radii = [float(radius_by_id[value]) for value in boundary]
        for point_id, value in zip(unknown, solved):
            original = float(radius_by_id[point_id])
            corrected.loc[corrected["id"].eq(point_id), "radius"] = float(value)
            events.append(
                {
                    "raw_point_id": point_id,
                    "raw_radius_um": original,
                    "normalized_radius_um": float(value),
                    "nonpositive_component_size": len(unknown),
                    "positive_boundary_point_ids": boundary,
                    "positive_boundary_radii_um": boundary_radii,
                    "normalization_rule": (
                        "topology_harmonic_interpolation_from_positive_neighbors"
                    ),
                }
            )
    if not corrected["radius"].gt(0).all():
        raise AssertionError("Radius normalization left a nonpositive value.")
    positive = dataframe["radius"].gt(0)
    if not np.array_equal(
        dataframe.loc[positive, "radius"].to_numpy(),
        corrected.loc[positive, "radius"].to_numpy(),
    ):
        raise AssertionError("A positive source radius changed during normalization.")
    return corrected, pd.DataFrame(events, columns=columns)


def _tree_path_length_um(dataframe: pd.DataFrame) -> float:
    """Return total parent-child Euclidean path length in micrometres."""

    by_id = dataframe.set_index("id")[["x", "y", "z"]]
    import math

    lengths: list[float] = []
    for row in dataframe.itertuples(index=False):
        if int(row.parent) == -1:
            continue
        child = np.asarray([row.x, row.y, row.z], dtype=float)
        parent = by_id.loc[int(row.parent)].to_numpy(dtype=float)
        lengths.append(float(np.linalg.norm(child - parent)))
    return math.fsum(sorted(lengths))


def create_simulation_morphologies(
    final_manifest: pd.DataFrame,
    project_root: str | Path,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create normalized CAVE-derived SWCs with auditable radius correction."""

    require_columns(
        final_manifest,
        {"model_node_id", "nucleus_id", "pt_root_id", "cave_skeleton_file"},
        "final manifest",
    )
    radius_policy = config["morphology"]["radius_normalization"]
    if radius_policy["rule"] != "topology_harmonic_interpolation_from_positive_neighbors":
        raise ValueError("Unsupported radius normalization rule.")
    output_dir = configured_path(project_root, config, "processed_morphologies")
    manifest_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    point_tables: list[pd.DataFrame] = []
    transition_tables: list[pd.DataFrame] = []
    radius_tables: list[pd.DataFrame] = []

    for row in final_manifest.sort_values("model_node_id").itertuples(index=False):
        raw_path = Path(project_root) / str(row.cave_skeleton_file)
        raw = read_swc(raw_path)
        raw_qc = validate_swc_tree(raw)
        if not raw_qc["finite_radii"]:
            raise ValueError(f"Raw skeleton has non-finite radii: {raw_path}")
        corrected, radius_events = normalize_nonpositive_radii(raw)
        normalized, point_mapping = normalize_swc(corrected)
        filename = (
            f"model_{int(row.model_node_id):03d}_"
            f"nucleus_{int(row.nucleus_id)}_root_{int(row.pt_root_id)}.swc"
        )
        target = output_dir / filename
        write_text_once(swc_text(normalized), target)
        reread = read_swc(target)

        ordered_raw = raw.set_index("id").loc[
            point_mapping.sort_values("simulation_point_id")["raw_point_id"]
        ]
        ordered_new = reread.sort_values("id").reset_index(drop=True)
        for column in ("type", "x", "y", "z"):
            if not np.array_equal(
                ordered_raw[column].to_numpy(), ordered_new[column].to_numpy()
            ):
                raise ValueError(f"{column} changed during normalization: {target}")
        positive = ordered_raw["radius"].gt(0).to_numpy()
        if not np.array_equal(
            ordered_raw.loc[positive, "radius"].to_numpy(),
            ordered_new.loc[positive, "radius"].to_numpy(),
        ):
            raise ValueError(f"A positive radius changed: {target}")
        raw_length = _tree_path_length_um(raw)
        simulation_length = _tree_path_length_um(reread)
        if raw_length != simulation_length:
            raise ValueError(f"Tree path length changed: {target}")

        morphology_qc = validate_simulation_morphology(target)
        if not morphology_qc["simulation_morphology_valid"]:
            raise ValueError(f"Simulation morphology failed QC: {target}")
        relative = str(target.resolve().relative_to(Path(project_root).resolve()))
        manifest_rows.append(
            {
                "model_node_id": int(row.model_node_id),
                "nucleus_id": int(row.nucleus_id),
                "pt_root_id": int(row.pt_root_id),
                "source_cave_skeleton": str(row.cave_skeleton_file),
                "simulation_morphology": relative,
                "sha256": sha256_file(target),
            }
        )
        qc_rows.append(
            {
                "model_node_id": int(row.model_node_id),
                "nucleus_id": int(row.nucleus_id),
                "pt_root_id": int(row.pt_root_id),
                "simulation_morphology": relative,
                "raw_tree_path_length_um": raw_length,
                "simulation_tree_path_length_um": simulation_length,
                "n_radius_normalized_points": int(len(radius_events)),
                **morphology_qc,
            }
        )
        point_mapping.insert(0, "pt_root_id", int(row.pt_root_id))
        point_mapping.insert(0, "model_node_id", int(row.model_node_id))
        point_tables.append(point_mapping)
        transitions = compartment_transition_events(normalized)
        if not transitions.empty:
            transitions.insert(0, "pt_root_id", int(row.pt_root_id))
            transitions.insert(0, "model_node_id", int(row.model_node_id))
            transition_tables.append(transitions)
        if not radius_events.empty:
            radius_events.insert(0, "pt_root_id", int(row.pt_root_id))
            radius_events.insert(0, "model_node_id", int(row.model_node_id))
            radius_tables.append(radius_events)

    transitions = pd.concat(transition_tables, ignore_index=True) if transition_tables else pd.DataFrame()
    radius_events = pd.concat(radius_tables, ignore_index=True) if radius_tables else pd.DataFrame()
    return (
        pd.DataFrame(manifest_rows).sort_values("model_node_id").reset_index(drop=True),
        pd.DataFrame(qc_rows).sort_values("model_node_id").reset_index(drop=True),
        pd.concat(point_tables, ignore_index=True),
        transitions,
        radius_events,
    )



def run_stage02(project_root: str | Path | None = None) -> dict[str, Any]:
    """Evaluate canonical cached CAVE skeleton biological eligibility."""

    from microns20 import cave
    from microns20.artifacts import write_dataframe
    from microns20.orchestration import artifact_path, project_context
    from microns20.provenance import require_completed_stage, write_stage_provenance

    root, config = project_context(project_root)
    require_completed_stage("01_candidate_discovery", root, config)
    candidates_path = artifact_path(root, config, "candidates", "biological_candidates.parquet")
    candidates = pd.read_parquet(candidates_path)
    eligible, qc, failures = evaluate_candidate_skeletons(None, candidates, root, config)
    output = artifact_path(root, config, "candidates", "morphology_eligible_candidates.parquet")
    qc_output = artifact_path(root, config, "results_tables", "cave_morphology_qc.parquet")
    failure_output = artifact_path(root, config, "results_tables", "cave_morphology_failures.parquet")
    for dataframe, path in [(eligible, output), (qc, qc_output), (failures, failure_output)]:
        write_dataframe(dataframe, path, overwrite=True)
    client = cave.create_client(config)
    provenance = write_stage_provenance(
        "02_cave_morphology_eligibility", root, config,
        inputs=[candidates_path], outputs=[output, qc_output, failure_output],
        source_metadata=cave.source_metadata(client, config),
        summaries={
            "n_candidate_rows": int(len(candidates)),
            "n_candidate_neurons": int(candidates["nucleus_id"].nunique()),
            "n_morphology_eligible_rows": int(len(eligible)),
            "n_morphology_eligible_neurons": int(eligible["nucleus_id"].nunique()),
            "n_morphology_failed_neurons": int(qc.loc[~qc["cave_morphology_eligible"], "nucleus_id"].nunique()),
            "n_neurons_requiring_radius_normalization": int(qc["requires_radius_normalization"].fillna(False).sum()),
            "n_nonpositive_radius_points": int(qc["n_nonpositive_radius_points"].fillna(0).sum()),
            "skeleton_service_requests": 0,
            "radius_policy": "defer finite nonpositive-radius normalization to Stage 07",
        },
    )
    return {"eligible_candidates": eligible, "morphology_qc": qc, "failures": failures, "provenance": provenance}


def run_stage07(project_root: str | Path | None = None) -> dict[str, Any]:
    """Create normalized simulation SWCs directly from frozen CAVE skeletons."""

    from microns20.artifacts import dataframe_digest, write_dataframe
    from microns20.config import configured_path
    from microns20.orchestration import artifact_path, project_context
    from microns20.provenance import require_completed_stage, write_stage_provenance
    from microns20.validation import validate_frozen_manifest

    root, config = project_context(project_root)
    require_completed_stage("06_functional_activity", root, config)
    manifest_path = root / "data/processed/final20_manifest.parquet"
    mapping_path = root / "data/processed/final20_functional_mappings.parquet"
    manifest = pd.read_parquet(manifest_path)
    mappings = pd.read_parquet(mapping_path)
    validate_frozen_manifest(manifest, mappings, config)
    morphology_manifest, morphology_qc, point_mapping, transitions, radius_events = create_simulation_morphologies(manifest, root, config)
    expected = manifest[["model_node_id", "nucleus_id", "pt_root_id"]]
    observed = morphology_manifest[["model_node_id", "nucleus_id", "pt_root_id"]]
    if dataframe_digest(expected) != dataframe_digest(observed):
        raise RuntimeError("Stage 07 changed frozen identities.")
    if len(morphology_manifest) != int(config["selection"]["n_neurons"]) or not morphology_qc["simulation_morphology_valid"].all():
        raise RuntimeError("Stage 07 simulation morphology validation failed.")
    manifest_output = configured_path(root, config, "processed_morphologies") / "manifest.parquet"
    qc_output = artifact_path(root, config, "results_tables", "simulation_morphology_qc.parquet")
    point_output = artifact_path(root, config, "morphologies", "cave_to_simulation_point_map.parquet")
    transition_output = artifact_path(root, config, "results_tables", "compartment_transition_events.parquet")
    radius_output = artifact_path(root, config, "results_tables", "radius_normalization_events.parquet")
    for dataframe, path in [(morphology_manifest, manifest_output), (morphology_qc, qc_output), (point_mapping, point_output), (transitions, transition_output), (radius_events, radius_output)]:
        write_dataframe(dataframe, path, overwrite=True)
    morphology_paths = [root / value for value in morphology_manifest["simulation_morphology"]]
    provenance = write_stage_provenance(
        "07_simulation_morphologies", root, config,
        inputs=[manifest_path, *[root / value for value in manifest["cave_skeleton_file"]]],
        outputs=[manifest_output, qc_output, point_output, transition_output, radius_output, *morphology_paths],
        source_metadata={
            "source": "CAVE skeleton service SWC",
            "coordinate_space": "MICrONS CAVE skeleton",
            "coordinate_unit": "micrometre",
            "materialization_version": int(config["cave"]["materialization_version"]),
            "skeleton_version": int(config["cave"]["skeleton_version"]),
        },
        summaries={
            "n_simulation_morphologies": int(len(morphology_manifest)),
            "n_valid_simulation_morphologies": int(morphology_qc["simulation_morphology_valid"].sum()),
            "n_radius_normalized_neurons": int(morphology_qc["n_radius_normalized_points"].gt(0).sum()),
            "n_radius_normalized_points": int(len(radius_events)),
            "radius_normalization_rule": config["morphology"]["radius_normalization"]["rule"],
            "coordinates_changed": False,
            "compartment_types_changed": False,
            "topology_changed_except_identifier_normalization": False,
        },
    )
    return {"morphology_manifest": morphology_manifest, "morphology_qc": morphology_qc, "point_mapping": point_mapping, "transitions": transitions, "radius_events": radius_events, "provenance": provenance}
