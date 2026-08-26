"""Functional identity validation with explicit deferred trace acquisition."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from microns20.artifacts import require_columns


def validate_manual_coregistration_mappings(manifest: pd.DataFrame, mappings: pd.DataFrame) -> pd.DataFrame:
    """Validate every frozen CAVE manual mapping without resolving a trace ROI."""

    manifest_columns = {"model_node_id", "nucleus_id", "pt_root_id", "pt_supervoxel_id", "session", "scan_idx", "n_functional_mappings"}
    mapping_columns = {"model_node_id", "nucleus_id", "pt_root_id", "pt_supervoxel_id", "session", "scan_idx", "unit_id", "field"}
    require_columns(manifest, manifest_columns, "final manifest")
    require_columns(mappings, mapping_columns, "final functional mappings")
    keys = ["nucleus_id", "session", "scan_idx", "unit_id", "field"]
    if mappings.duplicated(keys).any():
        raise ValueError("Frozen manual functional mapping keys are duplicated.")
    for column in ("model_node_id", "nucleus_id", "pt_root_id", "pt_supervoxel_id", "session", "scan_idx", "unit_id", "field"):
        if mappings[column].isna().any() or mappings[column].astype("int64").lt(0).any():
            raise ValueError(f"Functional mapping column {column} contains invalid IDs.")
    for column in ("nucleus_id", "pt_root_id", "pt_supervoxel_id"):
        if mappings[column].astype("int64").le(0).any():
            raise ValueError(f"Functional mapping column {column} must be positive.")
    identity_columns = ["model_node_id", "nucleus_id", "pt_root_id", "pt_supervoxel_id", "session", "scan_idx"]
    identity = manifest[identity_columns + ["n_functional_mappings"]]
    joined = mappings.merge(identity, on=identity_columns, how="left", validate="many_to_one", indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError("Functional mappings contain identities outside the frozen manifest.")
    counts = joined.groupby("model_node_id").size().rename("observed_mapping_count").reset_index()
    expected = manifest[["model_node_id", "n_functional_mappings"]].merge(counts, on="model_node_id", how="left", validate="one_to_one")
    if expected["observed_mapping_count"].isna().any() or not expected["observed_mapping_count"].eq(expected["n_functional_mappings"]).all():
        raise ValueError("Functional mapping multiplicity differs from the frozen manifest.")
    qc = mappings.copy()
    qc["functional_identity_source"] = "CAVE coregistration_manual_v4"
    qc["manual_mapping_valid"] = True
    qc["trace_acquisition_status"] = "DEFERRED_POST_SONATA"
    qc["trace_source_planned"] = "MICrONS NDA v8"
    return qc.sort_values(["model_node_id", "unit_id", "field"], kind="mergesort").reset_index(drop=True)


def deferred_trace_status(manifest: pd.DataFrame, mappings: pd.DataFrame, config: Mapping[str, Any]) -> dict[str, Any]:
    """Describe the intentional post-SONATA trace-acquisition boundary."""

    qc = validate_manual_coregistration_mappings(manifest, mappings)
    return {
        "status": "DEFERRED_POST_SONATA",
        "functional_identity_source": str(config["functional"]["identity_source"]),
        "planned_trace_source": str(config["functional"]["planned_trace_source"]),
        "structural_selection_gated_by_trace_availability": False,
        "n_frozen_neurons": int(manifest["nucleus_id"].nunique()),
        "n_preserved_functional_mappings": int(len(qc)),
        "n_multi_mapping_neurons": int(mappings.groupby("model_node_id").size().gt(1).sum()),
        "empty_trace_artifacts_written": False,
        "note": "CAVE manual coregistration is sufficient functional identity. Raw fluorescence and deconvolved activity will be acquired from MICrONS NDA v8 after structural SONATA construction.",
    }


def run_stage06(project_root: str | Path | None = None) -> dict[str, Any]:
    """Validate frozen mappings and record deferred NDA-v8 trace acquisition."""

    from microns20.artifacts import write_dataframe, write_json
    from microns20.config import configured_path
    from microns20.orchestration import artifact_path, project_context
    from microns20.provenance import require_completed_stage, write_stage_provenance
    from microns20.validation import validate_frozen_manifest

    root, config = project_context(project_root)
    require_completed_stage("05_final_qc_and_manifest", root, config)
    manifest_path = root / "data/processed/final20_manifest.parquet"
    mapping_path = root / "data/processed/final20_functional_mappings.parquet"
    manifest = pd.read_parquet(manifest_path)
    mappings = pd.read_parquet(mapping_path)
    validate_frozen_manifest(manifest, mappings, config)
    mapping_qc = validate_manual_coregistration_mappings(manifest, mappings)
    status = deferred_trace_status(manifest, mappings, config)
    qc_output = artifact_path(root, config, "results_tables", "final20_functional_mapping_qc.parquet")
    status_output = configured_path(root, config, "processed_functional") / "functional_trace_acquisition_status.json"
    write_dataframe(mapping_qc, qc_output, overwrite=True)
    write_json(status, status_output, overwrite=True)
    provenance = write_stage_provenance(
        "06_functional_activity", root, config,
        inputs=[manifest_path, mapping_path], outputs=[qc_output, status_output],
        summaries={**status, "functional_mapping_complete": True, "functional_trace_acquisition_deferred": True},
    )
    return {"mapping_qc": mapping_qc, "trace_status": status, "provenance": provenance}
