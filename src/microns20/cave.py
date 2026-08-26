"""Pinned CAVE access, immutable table snapshots, and synapse queries."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from caveclient import CAVEclient
import pandas as pd

from microns20.artifacts import require_columns, write_raw_snapshot
from microns20.config import cave_raw_directory


SYNAPSE_ID_COLUMNS = {
    "id",
    "pre_pt_supervoxel_id",
    "pre_pt_root_id",
    "post_pt_supervoxel_id",
    "post_pt_root_id",
}


def create_client(config: Mapping[str, Any]) -> CAVEclient:
    """Create a CAVE client pinned to the configured materialization."""

    cave = config["cave"]
    client = CAVEclient(str(cave["datastack"]))
    version = int(cave["materialization_version"])
    if version not in client.materialize.get_versions():
        raise RuntimeError(
            f"CAVE materialization {version} is unavailable for "
            f"{cave['datastack']}."
        )
    client.materialize.version = version
    return client


def materialization_timestamp(
    client: CAVEclient,
    config: Mapping[str, Any],
) -> datetime:
    """Return the exact configured materialization timestamp."""

    version = int(config["cave"]["materialization_version"])
    return client.materialize.get_timestamp(version=version)


def table_catalog(
    client: CAVEclient,
    config: Mapping[str, Any],
) -> list[str]:
    """List tables available in the configured materialization."""

    version = int(config["cave"]["materialization_version"])
    return sorted(client.materialize.get_tables(version=version))


def check_configured_tables(
    client: CAVEclient,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Validate configured table names and record live metadata."""

    available = set(table_catalog(client, config))
    rows = []
    for role, table_name in config["cave"]["tables"].items():
        required = role not in {"synapse_target_structure"}
        exists = str(table_name) in available if table_name else False
        if required and not exists:
            raise RuntimeError(
                f"Configured CAVE table for {role} is unavailable: {table_name}"
            )
        metadata: dict[str, Any] = {}
        if exists:
            metadata = client.materialize.get_table_metadata(
                str(table_name),
                version=int(config["cave"]["materialization_version"]),
            )
        rows.append(
            {
                "role": str(role),
                "table_name": str(table_name) if table_name else None,
                "required": bool(required),
                "available": bool(exists),
                "schema": metadata.get("schema"),
                "description": metadata.get("description"),
                "created": metadata.get("created"),
                "last_modified": metadata.get("last_modified"),
            }
        )

    advertised = client.info.get_datastack_info().get("synapse_table")
    configured = str(config["cave"]["tables"]["synapses"])
    if advertised != configured:
        raise RuntimeError(
            f"Configured synapse table {configured} differs from the "
            f"datastack-advertised table {advertised}."
        )
    return pd.DataFrame(rows)


def query_table(
    client: CAVEclient,
    table_name: str,
    config: Mapping[str, Any],
    *,
    filter_in_dict: Mapping[str, Iterable[Any]] | None = None,
    select_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Query a table explicitly at the configured materialization."""

    dataframe = client.materialize.query_table(
        table_name,
        filter_in_dict=dict(filter_in_dict) if filter_in_dict else None,
        select_columns=select_columns,
        materialization_version=int(
            config["cave"]["materialization_version"]
        ),
        split_positions=False,
    )
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f"CAVE query did not return a dataframe for {table_name}.")
    return dataframe


def raw_table_path(
    project_root: str | Path,
    config: Mapping[str, Any],
    table_name: str,
) -> Path:
    """Return the immutable path for one materialized CAVE table."""

    return cave_raw_directory(project_root, config) / "tables" / f"{table_name}.parquet"


def snapshot_table(
    client: CAVEclient,
    project_root: str | Path,
    config: Mapping[str, Any],
    table_name: str,
) -> tuple[pd.DataFrame, Path]:
    """Query and immutably snapshot a full pinned CAVE table."""

    incoming = query_table(client, table_name, config)
    target = raw_table_path(project_root, config, table_name)
    if target.exists():
        existing = pd.read_parquet(target)
        if set(existing.columns) == set(incoming.columns):
            incoming = incoming.reindex(columns=existing.columns)
    write_raw_snapshot(incoming, target)
    return pd.read_parquet(target), target


def _positive_unique(values: Iterable[int] | None) -> list[int] | None:
    if values is None:
        return None
    unique = sorted({int(value) for value in values if int(value) > 0})
    return unique


def validate_synapse_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Require stable IDs and fail if duplicate synapse IDs disagree."""

    require_columns(dataframe, SYNAPSE_ID_COLUMNS, "CAVE synapses")
    if dataframe.empty:
        return dataframe.copy()
    if dataframe["id"].isna().any():
        raise ValueError("CAVE synapses contain missing authoritative IDs.")

    duplicate_ids = dataframe["id"].duplicated(keep=False)
    if duplicate_ids.any():
        duplicate_rows = dataframe.loc[duplicate_ids]
        scalar_columns = [
            column
            for column in dataframe.columns
            if not duplicate_rows[column]
            .map(lambda value: hasattr(value, "shape"))
            .any()
        ]
        conflicts = (
            duplicate_rows.groupby("id")[scalar_columns]
            .nunique(dropna=False)
            .gt(1)
            .any(axis=1)
        )
        if conflicts.any():
            raise ValueError(
                "Duplicate CAVE synapse IDs disagree: "
                f"{conflicts[conflicts].index.astype(int).tolist()}"
            )
        dataframe = dataframe.drop_duplicates("id", keep="first")

    return dataframe.sort_values("id").reset_index(drop=True)


def query_synapses(
    client: CAVEclient,
    config: Mapping[str, Any],
    *,
    pre_ids: Iterable[int] | None = None,
    post_ids: Iterable[int] | None = None,
    page_size: int = 200_000,
    include_zeros: bool | None = None,
) -> pd.DataFrame:
    """Query pinned CAVE synapses with deterministic pagination."""

    pre = _positive_unique(pre_ids)
    post = _positive_unique(post_ids)
    if pre_ids is not None and not pre:
        return pd.DataFrame()
    if post_ids is not None and not post:
        return pd.DataFrame()
    if pre is None and post is None:
        raise ValueError("At least one of pre_ids or post_ids must be provided.")

    cave = config["cave"]
    pages = []
    offset = 0
    while True:
        page = client.materialize.synapse_query(
            pre_ids=pre,
            post_ids=post,
            remove_autapses=bool(config["synapses"]["remove_autapses"]),
            include_zeros=(
                bool(config["synapses"]["include_zero_roots"])
                if include_zeros is None else bool(include_zeros)
            ),
            limit=int(page_size),
            offset=int(offset),
            desired_resolution=list(cave["desired_resolution_um"]),
            materialization_version=int(cave["materialization_version"]),
            synapse_table=str(cave["tables"]["synapses"]),
        )
        if not isinstance(page, pd.DataFrame):
            raise TypeError("CAVE synapse query did not return a dataframe.")
        pages.append(page)
        if len(page) < page_size:
            break
        offset += len(page)

    if not pages or all(page.empty for page in pages):
        return pd.DataFrame()
    return validate_synapse_rows(pd.concat(pages, ignore_index=True))


def source_metadata(client: CAVEclient, config: Mapping[str, Any]) -> dict[str, Any]:
    """Return versioned CAVE source metadata for provenance."""

    version = int(config["cave"]["materialization_version"])
    metadata = client.materialize.get_version_metadata(version)
    return {
        "datastack": str(config["cave"]["datastack"]),
        "materialization_version": version,
        "materialization_timestamp": materialization_timestamp(
            client, config
        ).isoformat(),
        "materialization_status": metadata.get("status"),
        "skeleton_version": int(config["cave"]["skeleton_version"]),
        "synapse_table": str(config["cave"]["tables"]["synapses"]),
        "desired_resolution_um": list(
            config["cave"]["desired_resolution_um"]
        ),
    }


def run_stage00(project_root: str | Path | None = None) -> dict[str, Any]:
    """Run the CAVE-only structural source preflight."""

    from bmtk.builder.bionet import SWCReader
    import morphio
    from microns20 import morphology
    from microns20.artifacts import write_dataframe
    from microns20.orchestration import artifact_path, project_context
    from microns20.provenance import software_environment, write_stage_provenance
    from microns20.qc import path_preflight

    root, config = project_context(project_root)
    client = create_client(config)
    tables = check_configured_tables(client, config)
    manual = query_table(client, str(config["cave"]["tables"]["manual_coregistration"]), config)
    proof = query_table(client, str(config["cave"]["tables"]["proofreading"]), config)
    require_columns(manual, {"pt_root_id"}, "manual coregistration")
    require_columns(proof, {"pt_root_id", "status_axon", "status_dendrite"}, "proofreading")
    proof_roots = set(proof.loc[proof["status_axon"].fillna(False) & proof["status_dendrite"].fillna(False), "pt_root_id"].astype("int64"))
    roots = sorted(int(value) for value in manual["pt_root_id"].dropna().astype("int64").unique() if int(value) > 0 and int(value) in proof_roots)
    diagnostic_errors: list[str] = []
    skeleton_path: Path | None = None
    test_root: int | None = None
    for candidate_root in roots:
        candidate_path = morphology.raw_skeleton_path(root, config, candidate_root)
        if not candidate_path.is_file():
            continue
        try:
            tree = morphology.validate_swc_tree(morphology.read_swc(candidate_path))
            if not (tree["n_roots"] == 1 and tree["n_soma_points"] > 0 and tree["connected"] and tree["acyclic"]):
                raise ValueError("diagnostic SWC is not a soma-rooted tree")
            morphio.Morphology(str(candidate_path))
            reader = SWCReader(str(candidate_path))
            _ = reader.sections
            _ = reader.soma_position
            test_root = candidate_root
            skeleton_path = candidate_path
            break
        except Exception as error:
            diagnostic_errors.append(f"{candidate_root}: {type(error).__name__}: {error}")
    if skeleton_path is None or test_root is None:
        raise RuntimeError("No source-derived cached skeleton passed preflight: " + " | ".join(diagnostic_errors))
    rows = [
        {"check": "canonical_config", "status": "pass", "detail": str(root / "configs/project.yaml")},
        {"check": "cave_client", "status": "pass", "detail": str(config["cave"]["datastack"])},
        {"check": "cave_materialization", "status": "pass", "detail": str(config["cave"]["materialization_version"])},
        {"check": "cave_required_tables", "status": "pass", "detail": f"{int(tables['available'].sum())}/{len(tables)} available"},
        {"check": "cached_cave_skeleton", "status": "pass", "detail": f"dynamic root {test_root}; immutable cache; path={skeleton_path.relative_to(root)}"},
        {"check": "functional_identity", "status": "pass", "detail": "CAVE manual coregistration; all mappings preserved"},
        {"check": "functional_trace_acquisition", "status": "deferred", "detail": str(config["functional"]["planned_trace_source"])},
        {"check": "morphio_import", "status": "pass", "detail": str(skeleton_path.relative_to(root))},
        {"check": "bmtk_neuron_import", "status": "pass", "detail": str(skeleton_path.relative_to(root))},
    ]
    preflight = pd.concat([pd.DataFrame(rows), path_preflight(root, config)], ignore_index=True)
    output = artifact_path(root, config, "results_tables", "source_preflight.parquet")
    catalog_output = artifact_path(root, config, "results_tables", "cave_table_catalog.parquet")
    write_dataframe(preflight, output, overwrite=True)
    write_dataframe(tables, catalog_output, overwrite=True)
    provenance = write_stage_provenance(
        "00_source_preflight", root, config,
        inputs=[root / "configs/project.yaml", skeleton_path],
        outputs=[output, catalog_output],
        source_metadata=source_metadata(client, config),
        summaries={
            "functional_identity_source": config["functional"]["identity_source"],
            "functional_trace_acquisition_status": config["functional"]["trace_acquisition_status"],
            "n_cave_tables_checked": int(len(tables)),
            "dynamic_test_root": int(test_root),
            "skeleton_service_requests": 0,
            "software": software_environment(["caveclient", "pandas", "pyarrow", "scipy", "networkx", "MorphIO", "NEURON", "bmtk", "pcg-skel", "libsonata"]),
        },
    )
    return {"preflight": preflight, "table_catalog": tables, "provenance": provenance}
