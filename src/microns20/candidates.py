"""CAVE-only functional identity normalization and biological eligibility."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from microns20.artifacts import require_columns


FUNCTIONAL_MAPPING_COLUMNS = [
    "nucleus_id",
    "pt_root_id",
    "pt_supervoxel_id",
    "session",
    "scan_idx",
    "unit_id",
    "field",
    "coregistration_id",
    "id_ref",
    "score",
    "residual",
]


def _integer(series: pd.Series, name: str) -> pd.Series:
    converted = pd.to_numeric(series, errors="raise")
    if converted.isna().any():
        raise ValueError(f"{name} contains missing values.")
    return converted.astype("int64")


def normalize_functional_mappings(
    manual_coregistration: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Normalize manual coregistration without losing functional mappings."""

    required = {
        "id",
        "target_id",
        "pt_root_id",
        "pt_supervoxel_id",
        "session",
        "scan_idx",
        "unit_id",
        "field",
        "id_ref",
        "score",
        "residual",
    }
    require_columns(
        manual_coregistration,
        required,
        "manual coregistration",
    )

    renamed = manual_coregistration.rename(
        columns={
            "id": "coregistration_id",
            "target_id": "nucleus_id",
        }
    ).copy()
    for column in (
        "nucleus_id",
        "pt_root_id",
        "pt_supervoxel_id",
        "session",
        "scan_idx",
        "unit_id",
        "field",
        "coregistration_id",
        "id_ref",
    ):
        renamed[column] = _integer(renamed[column], column)

    valid_identity = (
        renamed["nucleus_id"].gt(0)
        & renamed["pt_root_id"].gt(0)
        & renamed["pt_supervoxel_id"].gt(0)
    )
    rejected = renamed.loc[~valid_identity, FUNCTIONAL_MAPPING_COLUMNS].copy()
    stable = renamed.loc[valid_identity, FUNCTIONAL_MAPPING_COLUMNS].copy()

    for column in ("pt_root_id", "pt_supervoxel_id"):
        ambiguous = stable.groupby("nucleus_id")[column].nunique().gt(1)
        if ambiguous.any():
            raise ValueError(
                f"Nuclei map to multiple {column} values: "
                f"{ambiguous[ambiguous].index.astype(int).tolist()}"
            )

    reverse = (
        stable.groupby("pt_supervoxel_id")["nucleus_id"].nunique().gt(1)
    )
    if reverse.any():
        raise ValueError(
            "Positive supervoxels map to multiple nuclei: "
            f"{reverse[reverse].index.astype(int).tolist()}"
        )

    mapping_key = [
        "nucleus_id",
        "session",
        "scan_idx",
        "unit_id",
        "field",
    ]
    repeated_key = stable.duplicated(mapping_key, keep=False)
    duplicate_report_rows = []
    if repeated_key.any():
        duplicate_rows = stable.loc[repeated_key]
        conflicts = (
            duplicate_rows.groupby(mapping_key, dropna=False)
            .nunique(dropna=False)
            .gt(1)
            .any(axis=1)
        )
        if conflicts.any():
            raise ValueError(
                "Manual functional mapping keys disagree: "
                f"{conflicts[conflicts].index.tolist()}"
            )

    exact_duplicate = stable.duplicated(FUNCTIONAL_MAPPING_COLUMNS, keep="first")
    for row in stable.loc[exact_duplicate].itertuples(index=False):
        duplicate_report_rows.append(
            {
                "nucleus_id": int(row.nucleus_id),
                "session": int(row.session),
                "scan_idx": int(row.scan_idx),
                "unit_id": int(row.unit_id),
                "field": int(row.field),
                "duplicate_kind": "exact",
            }
        )
    stable = stable.loc[~exact_duplicate].copy()

    if stable.duplicated(mapping_key).any():
        raise AssertionError("Functional mapping keys are not unique.")
    stable = stable.sort_values(mapping_key).reset_index(drop=True)
    duplicates = pd.DataFrame(
        duplicate_report_rows,
        columns=[
            "nucleus_id",
            "session",
            "scan_idx",
            "unit_id",
            "field",
            "duplicate_kind",
        ],
    )
    return stable, rejected.reset_index(drop=True), duplicates


def _unique_annotations(
    dataframe: pd.DataFrame,
    key: str,
    values: list[str],
    name: str,
) -> pd.DataFrame:
    require_columns(dataframe, {key, *values}, name)
    subset = dataframe[[key, *values]].copy()
    conflicts = (
        subset.groupby(key, dropna=False)[values]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )
    if conflicts.any():
        raise ValueError(
            f"{name} has conflicting annotations: "
            f"{conflicts[conflicts].index.tolist()}"
        )
    return subset.drop_duplicates(key)


def apply_reference_corrections(
    base: pd.DataFrame,
    corrections: pd.DataFrame,
    *,
    name: str,
) -> pd.DataFrame:
    """Apply explicit living-table corrections to a reference table."""

    values = ["classification_system", "cell_type"]
    base_unique = _unique_annotations(base, "target_id", values, name)
    if corrections.empty:
        return base_unique
    correction_unique = _unique_annotations(
        corrections,
        "target_id",
        values,
        f"{name} corrections",
    )
    unknown = set(correction_unique["target_id"]) - set(base_unique["target_id"])
    if unknown:
        raise ValueError(
            f"{name} corrections reference IDs absent from base: "
            f"{sorted(int(value) for value in unknown)}"
        )

    corrected = base_unique.set_index("target_id").copy()
    updates = correction_unique.set_index("target_id")
    corrected.loc[updates.index, values] = updates[values]
    return corrected.reset_index()


def build_biological_candidates(
    functional_mappings: pd.DataFrame,
    cell_types: pd.DataFrame,
    cell_type_corrections: pd.DataFrame,
    mtypes: pd.DataFrame,
    mtype_corrections: pd.DataFrame,
    functional_areas: pd.DataFrame,
    proofreading: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build and biologically filter one neuron-recording row."""

    require_columns(
        functional_mappings,
        FUNCTIONAL_MAPPING_COLUMNS,
        "functional mappings",
    )
    grouped = (
        functional_mappings.groupby(
            [
                "nucleus_id",
                "pt_root_id",
                "pt_supervoxel_id",
                "session",
                "scan_idx",
            ],
            as_index=False,
            sort=True,
        )
        .agg(n_functional_mappings=("unit_id", "size"))
    )

    cell = apply_reference_corrections(
        cell_types,
        cell_type_corrections,
        name="cell types",
    ).rename(
        columns={
            "target_id": "nucleus_id",
            "classification_system": "cell_class",
            "cell_type": "broad_cell_type",
        }
    )
    mtype = apply_reference_corrections(
        mtypes,
        mtype_corrections,
        name="mtypes",
    ).rename(
        columns={
            "target_id": "nucleus_id",
            "cell_type": "microns_mtype",
        }
    )[["nucleus_id", "microns_mtype"]]
    area = _unique_annotations(
        functional_areas,
        "target_id",
        ["tag"],
        "functional areas",
    ).rename(
        columns={"target_id": "nucleus_id", "tag": "functional_area"}
    )
    proof = _unique_annotations(
        proofreading,
        "pt_root_id",
        [
            "status_axon",
            "strategy_axon",
            "status_dendrite",
            "strategy_dendrite",
        ],
        "proofreading",
    )

    annotated = (
        grouped.merge(cell, on="nucleus_id", how="left", validate="many_to_one")
        .merge(mtype, on="nucleus_id", how="left", validate="many_to_one")
        .merge(area, on="nucleus_id", how="left", validate="many_to_one")
        .merge(proof, on="pt_root_id", how="left", validate="many_to_one")
    )

    rules = config["eligibility"]
    proof_rules = rules["proofreading"]
    masks = [
        ("manual_functional_coregistration", pd.Series(True, index=annotated.index)),
        (
            "configured_functional_area",
            annotated["functional_area"].eq(rules["functional_area"]),
        ),
        (
            "configured_cell_class",
            annotated["cell_class"].eq(rules["cell_class"]),
        ),
        (
            "proofreading_status",
            (
                annotated["status_axon"].fillna(False)
                if proof_rules["require_axon"]
                else pd.Series(True, index=annotated.index)
            )
            & (
                annotated["status_dendrite"].fillna(False)
                if proof_rules["require_dendrite"]
                else pd.Series(True, index=annotated.index)
            ),
        ),
        (
            "proofreading_strategy",
            annotated["strategy_axon"].isin(
                proof_rules["allowed_axon_strategies"]
            )
            & annotated["strategy_dendrite"].isin(
                proof_rules["allowed_dendrite_strategies"]
            ),
        ),
    ]

    cumulative = pd.Series(True, index=annotated.index)
    summary_rows = []
    for stage, mask in masks:
        cumulative &= mask.fillna(False)
        current = annotated.loc[cumulative]
        summary_rows.append(
            {
                "stage": stage,
                "n_candidate_rows": int(len(current)),
                "n_unique_neurons": int(current["nucleus_id"].nunique()),
                "n_recordings": int(
                    len(current[["session", "scan_idx"]].drop_duplicates())
                ),
            }
        )

    eligible = annotated.loc[cumulative].copy()
    eligible["biologically_eligible"] = True
    eligible = eligible.sort_values(
        ["session", "scan_idx", "nucleus_id"]
    ).reset_index(drop=True)
    failures = annotated.loc[~cumulative].copy()
    failures["biologically_eligible"] = False
    return eligible, pd.DataFrame(summary_rows), failures.reset_index(drop=True)


def functional_mappings_for_candidates(
    functional_mappings: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Return every normalized mapping belonging to candidate recordings."""

    keys = candidates[["nucleus_id", "session", "scan_idx"]].drop_duplicates()
    selected = functional_mappings.merge(
        keys,
        on=["nucleus_id", "session", "scan_idx"],
        how="inner",
        validate="many_to_one",
    )
    expected = int(candidates["n_functional_mappings"].sum())
    if len(selected) != expected:
        raise ValueError(
            f"Candidate mapping multiplicity changed: expected {expected}, "
            f"found {len(selected)}."
        )
    return selected.sort_values(
        ["session", "scan_idx", "nucleus_id", "unit_id", "field"]
    ).reset_index(drop=True)


def run_stage01(project_root: str | None = None) -> dict[str, Any]:
    """Build normalized functional mappings and biological candidates."""

    from microns20 import cave
    from microns20.artifacts import write_dataframe
    from microns20.orchestration import artifact_path, project_context
    from microns20.provenance import require_completed_stage, write_stage_provenance
    from microns20.validation import validate_stable_identity

    root, config = project_context(project_root)
    require_completed_stage("00_source_preflight", root, config)
    client = cave.create_client(config)
    roles = ["manual_coregistration", "functional_area", "proofreading", "cell_types", "cell_type_corrections", "mtypes", "mtype_corrections"]
    tables: dict[str, pd.DataFrame] = {}
    raw_paths = []
    for role in roles:
        table_name = str(config["cave"]["tables"][role])
        tables[role], raw_path = cave.snapshot_table(client, root, config, table_name)
        raw_paths.append(raw_path)
    mappings, rejected_identity, duplicates = normalize_functional_mappings(tables["manual_coregistration"])
    eligible, summary, failures = build_biological_candidates(
        mappings, tables["cell_types"], tables["cell_type_corrections"],
        tables["mtypes"], tables["mtype_corrections"], tables["functional_area"],
        tables["proofreading"], config,
    )
    validate_stable_identity(eligible)
    mapping_output = artifact_path(root, config, "candidates", "functional_mappings.parquet")
    candidate_output = artifact_path(root, config, "candidates", "biological_candidates.parquet")
    summary_output = artifact_path(root, config, "results_tables", "candidate_discovery_summary.parquet")
    failure_output = artifact_path(root, config, "results_tables", "candidate_discovery_failures.parquet")
    identity_output = artifact_path(root, config, "results_tables", "invalid_stable_identities.parquet")
    duplicate_output = artifact_path(root, config, "results_tables", "functional_mapping_duplicates.parquet")
    for dataframe, path in [(mappings, mapping_output), (eligible, candidate_output), (summary, summary_output), (failures, failure_output), (rejected_identity, identity_output), (duplicates, duplicate_output)]:
        write_dataframe(dataframe, path, overwrite=True)
    provenance = write_stage_provenance(
        "01_candidate_discovery", root, config, inputs=raw_paths,
        outputs=[mapping_output, candidate_output, summary_output, failure_output, identity_output, duplicate_output],
        source_metadata=cave.source_metadata(client, config),
        summaries={
            "n_normalized_functional_mappings": int(len(mappings)),
            "n_invalid_stable_identity_rows": int(len(rejected_identity)),
            "n_exact_duplicate_rows": int(len(duplicates)),
            "n_biologically_eligible_rows": int(len(eligible)),
            "n_biologically_eligible_neurons": int(eligible["nucleus_id"].nunique()),
            "filter_counts": summary.to_dict("records"),
        },
    )
    return {"functional_mappings": mappings, "biological_candidates": eligible, "summary": summary, "failures": failures, "invalid_identities": rejected_identity, "duplicates": duplicates, "provenance": provenance}
