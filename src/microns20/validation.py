"""Independent gates for stable identity, final freezing, and readiness."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bmtk.builder.bionet import SWCReader
from bmtk.utils.sonata.config import SonataConfig
import h5py
import libsonata
import morphio
import numpy as np
import pandas as pd

from microns20.artifacts import require_columns, sha256_file
from microns20.morphology import read_swc, validate_swc_tree


IDENTITY_COLUMNS = ["nucleus_id", "pt_root_id", "pt_supervoxel_id"]


def validate_stable_identity(dataframe: pd.DataFrame) -> None:
    """Require one-to-one positive biological, root, and supervoxel IDs."""

    require_columns(dataframe, IDENTITY_COLUMNS, "stable identity")
    unique = dataframe[IDENTITY_COLUMNS].drop_duplicates()
    for column in IDENTITY_COLUMNS:
        if unique[column].isna().any() or unique[column].astype("int64").le(0).any():
            raise ValueError(f"{column} must contain only positive IDs.")
        if unique[column].duplicated().any():
            raise ValueError(f"{column} is not unique across biological neurons.")


def assign_model_ids(
    population: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Assign deterministic contiguous model IDs only after selection."""

    if "model_node_id" in population.columns:
        raise ValueError("Population already has model_node_id.")
    sort_keys = list(config["selection"]["model_id_sort_keys"])
    require_columns(population, sort_keys, "model ID population")
    ordered = population.sort_values(sort_keys, kind="mergesort").reset_index(
        drop=True
    )
    start = int(config["selection"]["model_id_start"])
    ordered.insert(
        0,
        "model_node_id",
        range(start, start + len(ordered)),
    )
    return ordered


def independently_validate_skeletons(
    population: pd.DataFrame,
    project_root: str | Path,
) -> pd.DataFrame:
    """Reload and revalidate every selected canonical CAVE SWC."""

    require_columns(
        population,
        {
            "nucleus_id",
            "pt_root_id",
            "cave_skeleton_file",
            "cave_skeleton_sha256",
        },
        "selected population",
    )
    rows = []
    for row in population.itertuples(index=False):
        path = Path(project_root) / str(row.cave_skeleton_file)
        record = {
            "nucleus_id": int(row.nucleus_id),
            "pt_root_id": int(row.pt_root_id),
            "cave_skeleton_file": str(row.cave_skeleton_file),
            "independent_skeleton_valid": False,
            "independent_skeleton_error": "",
        }
        try:
            skeleton = read_swc(path)
            qc = validate_swc_tree(skeleton)
            soma = skeleton.loc[skeleton["type"].eq(1), ["x", "y", "z"]]
            if soma.empty:
                soma_xyz = [float("nan")] * 3
            else:
                soma_xyz = soma.to_numpy(dtype=float).mean(axis=0).tolist()
            record.update(dict(zip(["soma_x_um", "soma_y_um", "soma_z_um"], soma_xyz)))
            valid = bool(
                qc["n_roots"] == 1
                and qc["n_soma_points"] > 0
                and qc["n_axon_points"] > 0
                and qc["n_dendrite_points"] > 0
                and qc["finite_coordinates"]
                and qc["finite_radii"]
                and qc["unique_point_ids"]
                and qc["valid_parent_references"]
                and qc["acyclic"]
                and qc["connected"]
                and qc["allowed_types"]
            )
            record.update(qc)
            record["independent_skeleton_valid"] = valid
        except Exception as error:
            record["independent_skeleton_error"] = (
                f"{type(error).__name__}: {error}"
            )
        rows.append(record)
    return pd.DataFrame(rows)


def validate_functional_multiplicity(
    population: pd.DataFrame,
    mappings: pd.DataFrame,
) -> None:
    """Require every preserved functional mapping and no unrelated mappings."""

    require_columns(
        population,
        {"nucleus_id", "session", "scan_idx", "n_functional_mappings"},
        "population",
    )
    require_columns(
        mappings,
        {
            "nucleus_id",
            "session",
            "scan_idx",
            "unit_id",
            "field",
        },
        "functional mappings",
    )
    expected_keys = population[
        ["nucleus_id", "session", "scan_idx"]
    ].drop_duplicates()
    observed_keys = mappings[
        ["nucleus_id", "session", "scan_idx"]
    ].drop_duplicates()
    compared = expected_keys.merge(
        observed_keys,
        on=["nucleus_id", "session", "scan_idx"],
        how="outer",
        indicator=True,
    )
    if not compared["_merge"].eq("both").all():
        raise ValueError("Functional mappings do not match population identities.")
    counts = (
        mappings.groupby(["nucleus_id", "session", "scan_idx"])
        .size()
        .rename("observed")
        .reset_index()
    )
    expected = population[
        [
            "nucleus_id",
            "session",
            "scan_idx",
            "n_functional_mappings",
        ]
    ].merge(
        counts,
        on=["nucleus_id", "session", "scan_idx"],
        how="left",
        validate="one_to_one",
    )
    if not expected["observed"].eq(expected["n_functional_mappings"]).all():
        raise ValueError("Functional mapping multiplicity changed.")
    if mappings.duplicated(
        ["nucleus_id", "session", "scan_idx", "unit_id", "field"]
    ).any():
        raise ValueError("Final functional mapping keys are duplicated.")


def freeze_final_population(
    selected_unfrozen: pd.DataFrame,
    selected_mappings: pd.DataFrame,
    independent_skeleton_qc: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Freeze the exact validated population and normalized functional mappings."""

    n_expected = int(config["selection"]["n_neurons"])
    if len(selected_unfrozen) != n_expected:
        raise ValueError(
            f"Final selection has {len(selected_unfrozen)} rows, "
            f"expected {n_expected}."
        )
    if selected_unfrozen["nucleus_id"].nunique() != n_expected:
        raise ValueError("Final selection does not have unique biological neurons.")
    validate_stable_identity(selected_unfrozen)
    if not selected_unfrozen["cave_morphology_eligible"].fillna(False).all():
        raise ValueError("Final selection contains CAVE-ineligible morphology.")
    if not independent_skeleton_qc[
        "independent_skeleton_valid"
    ].fillna(False).all():
        failed = independent_skeleton_qc.loc[
            ~independent_skeleton_qc["independent_skeleton_valid"].fillna(False)
        ]
        raise ValueError(
            "Independent final skeleton validation failed: "
            f"{failed[['nucleus_id', 'independent_skeleton_error']].to_dict('records')}"
        )
    validate_functional_multiplicity(selected_unfrozen, selected_mappings)

    soma_columns = ["nucleus_id", "soma_x_um", "soma_y_um", "soma_z_um"]
    soma = independent_skeleton_qc[soma_columns].copy()
    if soma["nucleus_id"].duplicated().any():
        raise ValueError("Independent skeleton QC duplicated a nucleus.")
    population = selected_unfrozen.drop(
        columns=[column for column in soma_columns[1:] if column in selected_unfrozen],
    ).merge(soma, on="nucleus_id", how="left", validate="one_to_one")
    if population[soma_columns[1:]].isna().any().any():
        raise ValueError("Independent CAVE soma coordinates are incomplete.")
    manifest = assign_model_ids(population, config)
    identity = manifest[
        ["model_node_id", "nucleus_id", "session", "scan_idx"]
    ]
    mappings = selected_mappings.merge(
        identity,
        on=["nucleus_id", "session", "scan_idx"],
        how="inner",
        validate="many_to_one",
    )
    mappings = mappings[
        ["model_node_id"]
        + [column for column in mappings.columns if column != "model_node_id"]
    ]
    validate_functional_multiplicity(manifest, mappings)
    return (
        manifest.sort_values("model_node_id").reset_index(drop=True),
        mappings.sort_values(
            ["model_node_id", "unit_id", "field"]
        ).reset_index(drop=True),
    )


def validate_frozen_manifest(
    manifest: pd.DataFrame,
    functional_mappings: pd.DataFrame,
    config: Mapping[str, Any],
) -> None:
    """Validate a persisted final manifest without relying on stage state."""

    n_expected = int(config["selection"]["n_neurons"])
    if len(manifest) != n_expected:
        raise ValueError(
            f"Frozen manifest size is {len(manifest)}, expected {n_expected}."
        )
    validate_stable_identity(manifest)
    require_columns(
        manifest,
        {
            "model_node_id",
            "cave_morphology_eligible",
        },
        "frozen manifest",
    )
    expected_ids = list(
        range(
            int(config["selection"]["model_id_start"]),
            int(config["selection"]["model_id_start"]) + n_expected,
        )
    )
    if manifest.sort_values("nucleus_id")["model_node_id"].tolist() != expected_ids:
        raise ValueError("Model IDs are not deterministic by nucleus_id.")
    if not manifest["cave_morphology_eligible"].all():
        raise ValueError("Frozen manifest contains morphology failure.")
    validate_functional_multiplicity(manifest, functional_mappings)


def identity_fingerprint(dataframe: pd.DataFrame) -> list[tuple[int, int, int, int]]:
    """Return stable model/biological identity tuples for cross-stage checks."""

    require_columns(
        dataframe,
        {"model_node_id", *IDENTITY_COLUMNS},
        "identity fingerprint",
    )
    return sorted(
        (
            int(row.model_node_id),
            int(row.nucleus_id),
            int(row.pt_root_id),
            int(row.pt_supervoxel_id),
        )
        for row in dataframe.itertuples(index=False)
    )


def validate_sonata_type_tables(
    selected_node_types: pd.DataFrame,
    external_node_types: pd.DataFrame,
    intrinsic_edge_types: pd.DataFrame,
    external_edge_types: pd.DataFrame,
    selected_population: str,
    external_population: str,
) -> None:
    """Validate SONATA population and morphology-coordinate type metadata."""

    intrinsic_population = f"{selected_population}_to_{selected_population}"
    external_edge_population = f"{external_population}_to_{selected_population}"

    expected = {
        "selected node types": (
            selected_node_types,
            "node_type_id",
            selected_population,
        ),
        "external node types": (
            external_node_types,
            "node_type_id",
            external_population,
        ),
        "intrinsic edge types": (
            intrinsic_edge_types,
            "edge_type_id",
            intrinsic_population,
        ),
        "external edge types": (
            external_edge_types,
            "edge_type_id",
            external_edge_population,
        ),
    }

    for label, (frame, id_column, population_name) in expected.items():
        require_columns(
            frame,
            {id_column, "population"},
            label,
        )
        observed_populations = set(frame["population"].astype(str))
        if observed_populations != {population_name}:
            raise ValueError(
                f"{label} population mismatch: {observed_populations} != "
                f"{{{population_name!r}}}."
            )

    require_columns(
        selected_node_types,
        {"model_type", "recenter"},
        "selected node types",
    )
    if not selected_node_types["model_type"].astype(str).eq("biophysical").all():
        raise ValueError("Selected SONATA nodes must be biophysical.")
    if not selected_node_types["recenter"].astype(int).eq(0).all():
        raise ValueError(
            "Selected morphologies preserve global MICrONS coordinates, "
            "so SONATA recenter must be 0."
        )

    require_columns(
        external_node_types,
        {"model_type"},
        "external node types",
    )
    if not external_node_types["model_type"].astype(str).eq("virtual").all():
        raise ValueError("External SONATA nodes must be virtual.")


def validate_structural_sonata(
    manifest: pd.DataFrame,
    functional_mappings: pd.DataFrame,
    morphology_manifest: pd.DataFrame,
    external_manifest: pd.DataFrame,
    intrinsic_synapses: pd.DataFrame,
    external_synapses: pd.DataFrame,
    project_root: str | Path,
    sonata_root: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently validate the complete structural SONATA deliverable."""

    from microns20.sonata import get_sonata_population_names

    root = Path(project_root).resolve()
    package = Path(sonata_root).resolve()
    validate_frozen_manifest(manifest, functional_mappings, config)

    names = get_sonata_population_names(config)
    selected_name = names["selected"]
    external_name = names["external"]
    intrinsic_edge_name = names["intrinsic_edges"]
    external_edge_name = names["external_edges"]

    expected_count = int(config["selection"]["n_neurons"])
    if len(morphology_manifest) != expected_count:
        raise ValueError("Morphology manifest must contain one row per neuron.")

    identity_columns = ["model_node_id", "nucleus_id", "pt_root_id"]
    expected_identity = (
        manifest[identity_columns]
        .sort_values("model_node_id")
        .reset_index(drop=True)
    )
    observed_identity = (
        morphology_manifest[identity_columns]
        .sort_values("model_node_id")
        .reset_index(drop=True)
    )
    if not expected_identity.equals(observed_identity):
        raise ValueError("Morphology identities differ from the selected population.")

    circuit = SonataConfig.from_json(str(package / "circuit_config.json"))
    if not circuit.validate() or not circuit.with_networks:
        raise ValueError("BMTK rejected the structural circuit configuration.")
    if len(circuit.nodes) != 2 or len(circuit.edges) != 2:
        raise ValueError("Circuit config must expose two node and two edge files.")

    selected_nodes_path = (
        package
        / "network/nodes"
        / f"{selected_name}_nodes.h5"
    )
    selected_node_types_path = (
        package
        / "network/nodes"
        / f"{selected_name}_node_types.csv"
    )
    external_nodes_path = (
        package
        / "network/nodes"
        / f"{external_name}_nodes.h5"
    )
    external_node_types_path = (
        package
        / "network/nodes"
        / f"{external_name}_node_types.csv"
    )
    intrinsic_edges_path = (
        package
        / "network/edges"
        / f"{intrinsic_edge_name}_edges.h5"
    )
    intrinsic_edge_types_path = (
        package
        / "network/edges"
        / f"{intrinsic_edge_name}_edge_types.csv"
    )
    external_edges_path = (
        package
        / "network/edges"
        / f"{external_edge_name}_edges.h5"
    )
    external_edge_types_path = (
        package
        / "network/edges"
        / f"{external_edge_name}_edge_types.csv"
    )

    selected_node_types = pd.read_csv(selected_node_types_path)
    external_node_types = pd.read_csv(external_node_types_path)
    intrinsic_edge_types = pd.read_csv(intrinsic_edge_types_path)
    external_edge_types = pd.read_csv(external_edge_types_path)

    validate_sonata_type_tables(
        selected_node_types,
        external_node_types,
        intrinsic_edge_types,
        external_edge_types,
        selected_name,
        external_name,
    )

    selected_population = libsonata.NodeStorage(
        str(selected_nodes_path)
    ).open_population(selected_name)
    external_population = libsonata.NodeStorage(
        str(external_nodes_path)
    ).open_population(external_name)
    intrinsic_population = libsonata.EdgeStorage(
        str(intrinsic_edges_path)
    ).open_population(intrinsic_edge_name)
    external_edge_population = libsonata.EdgeStorage(
        str(external_edges_path)
    ).open_population(external_edge_name)

    if len(selected_population) != expected_count:
        raise ValueError("libSONATA selected population size is incorrect.")
    if len(external_population) != len(external_manifest):
        raise ValueError("libSONATA external population size is incorrect.")
    if len(intrinsic_population) != len(intrinsic_synapses):
        raise ValueError("libSONATA intrinsic edge count is incorrect.")
    if len(external_edge_population) != len(external_synapses):
        raise ValueError("libSONATA external edge count is incorrect.")

    with h5py.File(selected_nodes_path, "r") as handle:
        population = handle[f"nodes/{selected_name}"]
        if "node_id" not in population:
            raise ValueError("Selected SONATA population lacks required node_id.")

        observed_node_ids = np.asarray(
            population["node_id"],
            dtype=np.int64,
        )
        expected_node_ids = (
            manifest
            .sort_values("model_node_id")["model_node_id"]
            .to_numpy(dtype=np.int64)
        )
        if not np.array_equal(observed_node_ids, expected_node_ids):
            raise ValueError("Selected SONATA node_id values changed.")

        group = population["0"]
        order = np.asarray(
            population["node_group_index"],
            dtype=np.int64,
        )
        for column in (
            "model_node_id",
            "nucleus_id",
            "pt_root_id",
            "pt_supervoxel_id",
        ):
            observed = np.asarray(
                group[column],
                dtype=np.int64,
            )[order]
            expected_values = (
                manifest
                .sort_values("model_node_id")[column]
                .to_numpy(dtype=np.int64)
            )
            if not np.array_equal(observed, expected_values):
                raise ValueError(
                    f"SONATA node identity mismatch in {column}."
                )

        morphology_names = np.asarray(
            group["morphology"].asstr()
        )[order].tolist()

    morphology_dir = package / "components/morphologies"
    if len(set(morphology_names)) != expected_count:
        raise ValueError("SONATA morphology names are not one-to-one.")

    section_counts: dict[int, int] = {}
    copied_hashes = []
    source_by_model = morphology_manifest.set_index("model_node_id")

    for model_node_id, name in enumerate(morphology_names):
        copied = morphology_dir / name
        source = root / str(
            source_by_model.loc[
                model_node_id,
                "simulation_morphology",
            ]
        )
        if (
            not copied.is_file()
            or sha256_file(copied) != sha256_file(source)
        ):
            raise ValueError(
                f"SONATA morphology hash mismatch for model {model_node_id}."
            )

        morphio.Morphology(str(copied))
        qc = validate_swc_tree(read_swc(copied))
        valid = bool(
            qc["n_roots"] == 1
            and qc["n_soma_points"] > 0
            and qc["n_axon_points"] > 0
            and qc["n_dendrite_points"] > 0
            and qc["finite_coordinates"]
            and qc["finite_radii"]
            and not qc["has_nonpositive_radius"]
            and qc["connected"]
            and qc["acyclic"]
        )
        if not valid:
            raise ValueError(
                f"SONATA morphology QC failed for model {model_node_id}."
            )

        section_counts[model_node_id] = len(
            SWCReader(str(copied)).sections
        )
        copied_hashes.append(sha256_file(copied))

    with h5py.File(external_nodes_path, "r") as handle:
        population = handle[f"nodes/{external_name}"]
        if "node_id" not in population:
            raise ValueError("External SONATA population lacks required node_id.")

        observed_external_ids = np.asarray(
            population["node_id"],
            dtype=np.int64,
        )
        expected_external_ids = (
            external_manifest
            .sort_values("external_node_id")["external_node_id"]
            .to_numpy(dtype=np.int64)
        )
        if not np.array_equal(observed_external_ids, expected_external_ids):
            raise ValueError("External SONATA node_id values changed.")

        order = np.asarray(
            population["node_group_index"],
            dtype=np.int64,
        )
        observed_external = np.asarray(
            population["0/pt_root_id"],
            dtype=np.int64,
        )[order]

    expected_external = (
        external_manifest
        .sort_values("external_node_id")["pt_root_id"]
        .to_numpy(dtype=np.int64)
    )
    if not np.array_equal(observed_external, expected_external):
        raise ValueError("External SONATA nodes do not match their manifest.")
    if (
        (observed_external <= 0).any()
        or len(np.unique(observed_external)) != len(observed_external)
    ):
        raise ValueError("External SONATA roots must be positive and unique.")

    def validate_edges(
        path: Path,
        population_name: str,
        source_population_name: str,
        target_population_name: str,
        source_count: int,
        expected_synapses: pd.DataFrame,
    ) -> None:
        with h5py.File(path, "r") as handle:
            population = handle[f"edges/{population_name}"]
            source = np.asarray(
                population["source_node_id"],
                dtype=np.int64,
            )
            target = np.asarray(
                population["target_node_id"],
                dtype=np.int64,
            )

            observed_source_population = (
                population["source_node_id"].attrs["node_population"]
            )
            observed_target_population = (
                population["target_node_id"].attrs["node_population"]
            )

            if observed_source_population != source_population_name:
                raise ValueError(
                    f"{population_name} source population is incorrect."
                )
            if observed_target_population != target_population_name:
                raise ValueError(
                    f"{population_name} target population is incorrect."
                )

            if (source < 0).any() or (source >= source_count).any():
                raise ValueError(
                    f"{population_name} source IDs are out of range."
                )
            if (target < 0).any() or (target >= expected_count).any():
                raise ValueError(
                    f"{population_name} target IDs are out of range."
                )

            group_index = np.asarray(
                population["edge_group_index"],
                dtype=np.int64,
            )
            group = population["0"]
            synapse_ids = np.asarray(
                group["synapse_id"],
                dtype=np.int64,
            )[group_index]
            expected_ids = expected_synapses["id"].to_numpy(dtype=np.int64)

            if (
                len(np.unique(synapse_ids)) != len(synapse_ids)
                or not np.array_equal(
                    np.sort(synapse_ids),
                    np.sort(expected_ids),
                )
            ):
                raise ValueError(
                    f"{population_name} synapse identities changed."
                )

            section_id = np.asarray(
                group["afferent_section_id"],
                dtype=np.int64,
            )[group_index]
            section_pos = np.asarray(
                group["afferent_section_pos"],
                dtype=float,
            )[group_index]
            if (
                not np.isfinite(section_pos).all()
                or ((section_pos < 0) | (section_pos > 1)).any()
            ):
                raise ValueError(
                    f"{population_name} has invalid section positions."
                )

            section_type = np.asarray(
                group["afferent_section_type"],
                dtype=np.int64,
            )[group_index]
            if not np.isin(section_type, [1, 2, 3, 4]).all():
                raise ValueError(
                    f"{population_name} has invalid section types."
                )

            limits = np.asarray(
                [section_counts[int(value)] for value in target]
            )
            if (
                (section_id < 0).any()
                or (section_id >= limits).any()
            ):
                raise ValueError(
                    f"{population_name} has invalid section IDs."
                )

            for attribute in (
                "afferent_center_x",
                "afferent_center_y",
                "afferent_center_z",
                "placement_distance",
                "source_vertex_distance",
                "synapse_size",
            ):
                values = np.asarray(
                    group[attribute],
                    dtype=float,
                )[group_index]
                if not np.isfinite(values).all():
                    raise ValueError(
                        f"{population_name} has non-finite {attribute}."
                    )

            if "indices" not in population:
                raise ValueError(
                    f"{population_name} lacks SONATA edge indices."
                )

    validate_edges(
        intrinsic_edges_path,
        intrinsic_edge_name,
        selected_name,
        selected_name,
        expected_count,
        intrinsic_synapses,
    )
    validate_edges(
        external_edges_path,
        external_edge_name,
        external_name,
        selected_name,
        len(external_manifest),
        external_synapses,
    )

    forbidden = {
        "syn_weight",
        "delay",
        "dynamics_params",
        "model_template",
    }
    for frame, label in (
        (intrinsic_edge_types, "intrinsic edge types"),
        (external_edge_types, "external edge types"),
    ):
        columns = set(frame.columns)
        if columns & forbidden:
            raise ValueError(
                f"Pre-e-model {label} invent physiology: "
                f"{columns & forbidden}"
            )

    return {
        "n_frozen_neurons": int(len(manifest)),
        "n_functional_mappings": int(len(functional_mappings)),
        "n_multi_mapping_neurons": int(
            functional_mappings
            .groupby("model_node_id")
            .size()
            .gt(1)
            .sum()
        ),
        "n_simulation_morphologies": int(len(copied_hashes)),
        "n_external_nodes": int(len(external_manifest)),
        "n_intrinsic_edges": int(len(intrinsic_synapses)),
        "n_external_incoming_edges": int(len(external_synapses)),
        "functional_mapping_complete": True,
        "functional_trace_acquisition_deferred": True,
        "structural_sonata_valid": True,
        "ready_for_emodel_attachment": True,
        "ready_for_functional_validation": False,
        "ready_for_biophysical_simulation": False,
        "physiology_assigned": False,
        "bmtk_config_valid": True,
        "libsonata_populations_loadable": True,
        "morphio_morphologies_loadable": True,
        "bmtk_morphologies_loadable": True,
        "identity_preserved": True,
        "synapse_identity_preserved": True,
        "sonata_node_ids_explicit": True,
        "sonata_type_populations_valid": True,
        "morphology_recenter_disabled": True,
        "selected_population": selected_name,
        "external_population": external_name,
    }


def run_stage05(project_root: str | Path | None = None) -> dict[str, Any]:
    """Independently validate and freeze the exact selected population."""

    from microns20 import candidates as candidate_logic, cave, connectivity
    from microns20.artifacts import dataframe_digest, write_dataframe
    from microns20.orchestration import archive_replaced_artifact, artifact_path, project_context
    from microns20.provenance import require_completed_stage, write_stage_provenance
    from microns20.qc import population_summary, soma_spatial_qc

    root, config = project_context(project_root)
    require_completed_stage("04_cave_connectivity_selection", root, config)
    selected_path = artifact_path(root, config, "selection", "selected_population_unfrozen.parquet")
    eligible_path = artifact_path(root, config, "candidates", "morphology_eligible_candidates.parquet")
    mapping_path = artifact_path(root, config, "recordings", "selected_recording_functional_mappings.parquet")
    all_mapping_path = artifact_path(root, config, "candidates", "functional_mappings.parquet")
    synapse_path = artifact_path(root, config, "recordings", "selected_recording_synapses.parquet")
    selected = pd.read_parquet(selected_path)
    eligible = pd.read_parquet(eligible_path)
    expected_mappings = pd.read_parquet(mapping_path)
    all_mappings = pd.read_parquet(all_mapping_path)
    expected_synapses = pd.read_parquet(synapse_path)
    eligible_keys = set(map(tuple, eligible[["nucleus_id", "session", "scan_idx"]].astype("int64").to_numpy()))
    selected_keys = set(map(tuple, selected[["nucleus_id", "session", "scan_idx"]].astype("int64").to_numpy()))
    if not selected_keys.issubset(eligible_keys):
        raise ValueError("Selected identities are not a subset of the eligible pool.")
    skeleton_qc = independently_validate_skeletons(selected, root)
    independent_mappings = candidate_logic.functional_mappings_for_candidates(all_mappings, selected)
    expected_mappings = candidate_logic.functional_mappings_for_candidates(expected_mappings, selected)
    columns = candidate_logic.FUNCTIONAL_MAPPING_COLUMNS
    if dataframe_digest(expected_mappings[columns]) != dataframe_digest(independent_mappings[columns]):
        raise RuntimeError("Independent functional mapping derivation disagrees with Stage 03.")
    client = cave.create_client(config)
    roots = selected["pt_root_id"].astype(int).tolist()
    expected_synapses = expected_synapses.loc[expected_synapses["pre_pt_root_id"].isin(roots) & expected_synapses["post_pt_root_id"].isin(roots)].copy()
    independent_synapses = cave.query_synapses(client, config, pre_ids=roots, post_ids=roots)
    synapse_comparison = connectivity.compare_synapse_snapshots(expected_synapses, independent_synapses)
    if not (synapse_comparison["_merge"].eq("both") & synapse_comparison["endpoints_agree"]).all():
        raise RuntimeError("Independent CAVE connectivity query disagrees.")
    manifest, final_mappings = freeze_final_population(selected, independent_mappings, skeleton_qc, config)
    manifest["skeleton_file"] = manifest["cave_skeleton_file"]
    manifest["skeleton_version"] = int(config["cave"]["skeleton_version"])
    manifest["functional_mapping_complete"] = True
    manifest["functional_trace_acquisition_deferred"] = True
    validate_frozen_manifest(manifest, final_mappings, config)
    spatial_table, spatial_summary = soma_spatial_qc(manifest)
    manifest_output = root / "data/processed/final20_manifest.parquet"
    mapping_output = root / "data/processed/final20_functional_mappings.parquet"
    archived = [value for value in [archive_replaced_artifact(manifest_output, root), archive_replaced_artifact(mapping_output, root)] if value is not None]
    skeleton_output = artifact_path(root, config, "results_tables", "final20_independent_skeleton_qc.parquet")
    connectivity_output = artifact_path(root, config, "results_tables", "final20_connectivity_crosscheck.parquet")
    spatial_output = artifact_path(root, config, "results_tables", "final20_spatial_qc.parquet")
    for dataframe, path in [(manifest, manifest_output), (final_mappings, mapping_output), (skeleton_qc, skeleton_output), (synapse_comparison, connectivity_output), (spatial_table, spatial_output)]:
        write_dataframe(dataframe, path, overwrite=True)
    provenance = write_stage_provenance(
        "05_final_qc_and_manifest", root, config,
        inputs=[selected_path, eligible_path, mapping_path, all_mapping_path, synapse_path],
        outputs=[manifest_output, mapping_output, skeleton_output, connectivity_output, spatial_output],
        source_metadata=cave.source_metadata(client, config),
        summaries={
            "population": population_summary(manifest),
            "n_functional_mappings": int(len(final_mappings)),
            "functional_mapping_complete": True,
            "functional_trace_acquisition_deferred": True,
            "n_internal_synapses": int(len(independent_synapses)),
            "spatial_qc": spatial_summary,
            "archived_pre_rebuild_artifacts": [str(path.relative_to(root)) for path in archived],
        },
    )
    return {"manifest": manifest, "functional_mappings": final_mappings, "skeleton_qc": skeleton_qc, "synapse_comparison": synapse_comparison, "spatial_qc": spatial_table, "spatial_summary": spatial_summary, "provenance": provenance}


def run_stage10(project_root: str | Path | None = None) -> dict[str, Any]:
    """Independently validate the end-to-end structural deliverable."""

    from microns20.artifacts import write_dataframe, write_json
    from microns20.config import configured_path
    from microns20.orchestration import artifact_path, project_context
    from microns20.provenance import require_completed_stage, write_stage_provenance

    root, config = project_context(project_root)
    require_completed_stage("09_structural_sonata", root, config)
    manifest_path = root / "data/processed/final20_manifest.parquet"
    mapping_path = root / "data/processed/final20_functional_mappings.parquet"
    morphology_manifest_path = configured_path(root, config, "processed_morphologies") / "manifest.parquet"
    connectivity_dir = configured_path(root, config, "processed_connectivity")
    external_manifest_path = connectivity_dir / "external_presynaptic_nodes.parquet"
    intrinsic_path = connectivity_dir / "intrinsic_synapses.parquet"
    incoming_path = connectivity_dir / "incoming_external_synapses.parquet"
    sonata_root = configured_path(root, config, "processed_sonata")
    manifest = pd.read_parquet(manifest_path)
    mappings = pd.read_parquet(mapping_path)
    morphology_manifest = pd.read_parquet(morphology_manifest_path)
    external_manifest = pd.read_parquet(external_manifest_path)
    intrinsic = pd.read_parquet(intrinsic_path)
    incoming = pd.read_parquet(incoming_path)
    readiness = validate_structural_sonata(manifest, mappings, morphology_manifest, external_manifest, intrinsic, incoming, root, sonata_root, config)
    table = pd.DataFrame([readiness])
    validation_output = artifact_path(root, config, "results_tables", "end_to_end_structural_validation.parquet")
    readiness_output = sonata_root / "structural_readiness.json"
    write_dataframe(table, validation_output, overwrite=True)
    write_json(readiness, readiness_output, overwrite=True)
    package_inputs = [path for path in sonata_root.rglob("*") if path.is_file()]
    provenance = write_stage_provenance(
        "10_end_to_end_validation", root, config,
        inputs=[manifest_path, mapping_path, morphology_manifest_path, external_manifest_path, intrinsic_path, incoming_path, *package_inputs],
        outputs=[validation_output, readiness_output],
        source_metadata={"validator": "independent HDF5, libSONATA, BMTK, MorphIO, and SWC reload", "functional_trace_source_status": "deferred_post_sonata"},
        summaries=readiness,
    )
    return {"readiness": readiness, "validation": table, "provenance": provenance}
