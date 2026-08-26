"""Audit the Stage 01 proofreading lineage without changing eligibility."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from microns20.artifacts import write_dataframe
from microns20.candidates import (
    apply_reference_corrections,
    normalize_functional_mappings,
)
from microns20.config import (
    cave_raw_directory,
    config_sha256,
    find_project_root,
    load_config,
)
from microns20.provenance import write_stage_provenance


def _unique_annotations(
    dataframe: pd.DataFrame,
    key: str,
    value_columns: list[str],
    name: str,
) -> pd.DataFrame:
    """Require one logical annotation per key."""

    subset = dataframe[[key, *value_columns]].copy()
    conflicts = (
        subset.groupby(key, dropna=False)[value_columns]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )
    if conflicts.any():
        raise ValueError(
            f"{name} has conflicting values for "
            f"{conflicts[conflicts].index.tolist()}."
        )
    return subset.drop_duplicates(key)


def _count(
    dataframe: pd.DataFrame,
    mask: pd.Series,
    *,
    neuron_column: str,
) -> tuple[int, int]:
    selected = dataframe.loc[mask.fillna(False)]
    return int(len(selected)), int(selected[neuron_column].nunique())


def _condition_rows(
    proof: pd.DataFrame,
) -> list[tuple[str, str, Callable[[pd.DataFrame], pd.Series]]]:
    conditions: list[
        tuple[str, str, Callable[[pd.DataFrame], pd.Series]]
    ] = [
        ("all", "all proofread table", lambda data: pd.Series(True, index=data.index)),
        (
            "status",
            "status_axon=True",
            lambda data: data["status_axon"].fillna(False),
        ),
        (
            "status",
            "status_dendrite=True",
            lambda data: data["status_dendrite"].fillna(False),
        ),
        (
            "status",
            "status_axon=True AND status_dendrite=True",
            lambda data: data["status_axon"].fillna(False)
            & data["status_dendrite"].fillna(False),
        ),
    ]
    for value in sorted(proof["strategy_axon"].astype(str).unique()):
        conditions.append(
            (
                "strategy_axon",
                f"strategy_axon={value}",
                lambda data, value=value: data["strategy_axon"].astype(str).eq(value),
            )
        )
    for value in sorted(proof["strategy_dendrite"].astype(str).unique()):
        conditions.append(
            (
                "strategy_dendrite",
                f"strategy_dendrite={value}",
                lambda data, value=value: data["strategy_dendrite"]
                .astype(str)
                .eq(value),
            )
        )
    conditions.append(
        (
            "joint_strategy",
            "axon_fully_extended AND dendrite_extended",
            lambda data: data["strategy_axon"].eq("axon_fully_extended")
            & data["strategy_dendrite"].eq("dendrite_extended"),
        )
    )
    return conditions


def main() -> None:
    """Create cumulative, non-cumulative, policy, and lineage diagnostics."""

    root = find_project_root()
    config = load_config(root)
    config_hash_before = config_sha256(root)
    table_dir = cave_raw_directory(root, config) / "tables"

    table_names = config["cave"]["tables"]
    input_paths = {
        "proof": table_dir / f"{table_names['proofreading']}.parquet",
        "manual": table_dir / f"{table_names['manual_coregistration']}.parquet",
        "area": table_dir / f"{table_names['functional_area']}.parquet",
        "cell": table_dir / f"{table_names['cell_types']}.parquet",
        "cell_corrections": (
            table_dir / f"{table_names['cell_type_corrections']}.parquet"
        ),
    }
    proof = pd.read_parquet(input_paths["proof"])
    manual_raw = pd.read_parquet(input_paths["manual"])
    area_raw = pd.read_parquet(input_paths["area"])
    cell_raw = pd.read_parquet(input_paths["cell"])
    cell_corrections = pd.read_parquet(input_paths["cell_corrections"])

    required_proof = {
        "pt_root_id",
        "status_axon",
        "status_dendrite",
        "strategy_axon",
        "strategy_dendrite",
    }
    missing = required_proof - set(proof.columns)
    if missing:
        raise ValueError(f"Proofreading table lacks {sorted(missing)}.")
    if proof["pt_root_id"].isna().any() or proof["pt_root_id"].le(0).any():
        raise ValueError("Proofreading table contains a missing/nonpositive root.")
    if proof["pt_root_id"].duplicated().any():
        raise ValueError("Proofreading table contains duplicate root IDs.")

    mappings, invalid_mappings, exact_duplicates = (
        normalize_functional_mappings(manual_raw)
    )
    neuron_recordings = (
        mappings.groupby(
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

    proof_columns = [
        "pt_root_id",
        "status_axon",
        "status_dendrite",
        "strategy_axon",
        "strategy_dendrite",
    ]
    joined = neuron_recordings.merge(
        proof[proof_columns],
        on="pt_root_id",
        how="inner",
        validate="many_to_one",
    )
    area = _unique_annotations(
        area_raw,
        "target_id",
        ["tag"],
        "functional areas",
    ).rename(columns={"target_id": "nucleus_id", "tag": "functional_area"})
    cell = apply_reference_corrections(
        cell_raw,
        cell_corrections,
        name="cell types",
    ).rename(
        columns={
            "target_id": "nucleus_id",
            "classification_system": "cell_class",
            "cell_type": "broad_cell_type",
        }
    )
    joined = (
        joined.merge(area, on="nucleus_id", how="left", validate="many_to_one")
        .merge(cell, on="nucleus_id", how="left", validate="many_to_one")
    )
    is_v1 = joined["functional_area"].eq(config["eligibility"]["functional_area"])
    is_excitatory = joined["cell_class"].eq(
        config["eligibility"]["cell_class"]
    )

    conditions = _condition_rows(proof)
    noncumulative_rows = []
    for group, label, condition in conditions:
        rows, neurons = _count(
            proof,
            condition(proof),
            neuron_column="pt_root_id",
        )
        noncumulative_rows.append(
            {
                "audit_group": group,
                "condition": label,
                "rows": rows,
                "unique_neurons": neurons,
                "neuron_identity": "pt_root_id",
                "row_semantics": "proofreading_table_row",
            }
        )
    noncumulative = pd.DataFrame(noncumulative_rows)

    intersection_rows = []
    for group, label, condition in conditions:
        raw_mask = condition(proof)
        rows, neurons = _count(
            proof, raw_mask, neuron_column="pt_root_id"
        )
        intersection_rows.append(
            {
                "audit_group": group,
                "condition": label,
                "intersection": "proofreading table",
                "rows": rows,
                "unique_neurons": neurons,
                "row_semantics": "proofreading_table_row",
            }
        )
        manual_mask = condition(joined)
        stages = [
            ("+ manual functional coregistration", manual_mask),
            ("+ V1", manual_mask & is_v1),
            ("+ excitatory", manual_mask & is_v1 & is_excitatory),
        ]
        for stage, mask in stages:
            rows, neurons = _count(
                joined, mask, neuron_column="nucleus_id"
            )
            intersection_rows.append(
                {
                    "audit_group": group,
                    "condition": label,
                    "intersection": stage,
                    "rows": rows,
                    "unique_neurons": neurons,
                    "row_semantics": "biological_neuron_recording",
                }
            )
    intersections = pd.DataFrame(intersection_rows)

    cumulative_rows = [
        {
            "filter": "all proofread table",
            "rows": int(len(proof)),
            "unique_neurons": int(proof["pt_root_id"].nunique()),
            "row_semantics": "proofreading_table_row",
        }
    ]
    cumulative_mask = pd.Series(True, index=joined.index)
    cumulative_steps = [
        ("manually functionally coregistered", pd.Series(True, index=joined.index)),
        ("+ V1", is_v1),
        ("+ excitatory", is_excitatory),
        ("+ status_axon=True", joined["status_axon"].fillna(False)),
        (
            "+ status_dendrite=True",
            joined["status_dendrite"].fillna(False),
        ),
        (
            "+ axon_fully_extended",
            joined["strategy_axon"].eq("axon_fully_extended"),
        ),
        (
            "+ dendrite_extended",
            joined["strategy_dendrite"].eq("dendrite_extended"),
        ),
    ]
    for label, mask in cumulative_steps:
        cumulative_mask &= mask.fillna(False)
        rows, neurons = _count(
            joined, cumulative_mask, neuron_column="nucleus_id"
        )
        cumulative_rows.append(
            {
                "filter": label,
                "rows": rows,
                "unique_neurons": neurons,
                "row_semantics": "biological_neuron_recording",
            }
        )
    cumulative = pd.DataFrame(cumulative_rows)

    policy_functions: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
        "Policy A": lambda data: data["strategy_axon"].eq(
            "axon_fully_extended"
        )
        & data["strategy_dendrite"].eq("dendrite_extended"),
        "Policy B": lambda data: data["status_axon"].fillna(False)
        & data["strategy_dendrite"].eq("dendrite_extended"),
        "Policy C": lambda data: data["status_axon"].fillna(False)
        & data["strategy_axon"].isin(
            ["axon_fully_extended", "axon_partially_extended"]
        )
        & data["strategy_dendrite"].eq("dendrite_extended"),
        "Policy D": lambda data: data["status_axon"].fillna(False)
        & data["status_dendrite"].fillna(False),
    }
    policy_descriptions = {
        "Policy A": "axon_fully_extended + dendrite_extended",
        "Policy B": "status_axon=True + dendrite_extended",
        "Policy C": (
            "(axon_fully_extended OR axon_partially_extended) + "
            "status_axon=True + dendrite_extended"
        ),
        "Policy D": "status_axon=True + status_dendrite=True",
    }
    all_recordings = (
        neuron_recordings[["session", "scan_idx"]]
        .drop_duplicates()
        .sort_values(["session", "scan_idx"])
    )
    policy_summary_rows = []
    policy_recording_tables = []
    for policy, policy_function in policy_functions.items():
        proof_mask = policy_function(proof)
        manual_mask = policy_function(joined)
        candidate_mask = manual_mask & is_v1 & is_excitatory
        manual_subset = joined.loc[manual_mask]
        candidate_subset = joined.loc[candidate_mask]

        manual_counts = (
            manual_subset.groupby(["session", "scan_idx"])
            .agg(n_manual_coreg_neurons=("nucleus_id", "nunique"))
            .reset_index()
        )
        candidate_counts = (
            candidate_subset.groupby(["session", "scan_idx"])
            .agg(
                n_manual_v1_excitatory_neurons=("nucleus_id", "nunique"),
                n_candidate_rows=("nucleus_id", "size"),
            )
            .reset_index()
        )
        recording_counts = (
            all_recordings.merge(
                manual_counts,
                on=["session", "scan_idx"],
                how="left",
                validate="one_to_one",
            )
            .merge(
                candidate_counts,
                on=["session", "scan_idx"],
                how="left",
                validate="one_to_one",
            )
        )
        count_columns = [
            "n_manual_coreg_neurons",
            "n_manual_v1_excitatory_neurons",
            "n_candidate_rows",
        ]
        recording_counts[count_columns] = (
            recording_counts[count_columns].fillna(0).astype("int64")
        )
        recording_counts.insert(0, "policy", policy)
        recording_counts.insert(
            1, "policy_definition", policy_descriptions[policy]
        )
        policy_recording_tables.append(recording_counts)
        policy_summary_rows.append(
            {
                "policy": policy,
                "policy_definition": policy_descriptions[policy],
                "total_unique_neurons": int(
                    proof.loc[proof_mask, "pt_root_id"].nunique()
                ),
                "unique_neurons_with_manual_functional_coregistration": int(
                    manual_subset["nucleus_id"].nunique()
                ),
                "unique_manual_v1_excitatory_neurons": int(
                    candidate_subset["nucleus_id"].nunique()
                ),
                "candidate_neuron_recording_rows": int(len(candidate_subset)),
                "recordings_ge_20_manual_coreg": int(
                    recording_counts["n_manual_coreg_neurons"].ge(20).sum()
                ),
                "recordings_ge_20_manual_v1_excitatory": int(
                    recording_counts[
                        "n_manual_v1_excitatory_neurons"
                    ].ge(20).sum()
                ),
            }
        )
    policy_summary = pd.DataFrame(policy_summary_rows)
    policy_recordings = pd.concat(
        policy_recording_tables, ignore_index=True
    )

    stage01_candidates_path = (
        root / "data/interim/candidates/biological_candidates.parquet"
    )
    stage01_candidates = pd.read_parquet(stage01_candidates_path)
    stage02_qc_path = root / "results/tables/cave_morphology_qc.parquet"
    stage02_qc = pd.read_parquet(stage02_qc_path)
    if len(stage02_qc) != stage01_candidates["nucleus_id"].nunique():
        raise ValueError("Stage 02 QC is not one row per Stage 01 neuron.")
    if not stage02_qc["skeleton_retrieved"].fillna(False).all():
        raise ValueError("Stage 02 did not retrieve every requested skeleton.")
    source_counts = stage02_qc["cave_skeleton_source"].value_counts()
    requests = int(len(stage02_qc))
    lineage = pd.DataFrame(
        [
            {
                "lineage_step": "fresh proofreading table",
                "rows": int(len(proof)),
                "unique_neurons": int(proof["pt_root_id"].nunique()),
                "unique_roots": int(proof["pt_root_id"].nunique()),
                "skeleton_service_requests": 0,
                "detail": "No skeleton access; immutable CAVE table snapshot.",
            },
            {
                "lineage_step": "normalized positive manual mappings",
                "rows": int(len(mappings)),
                "unique_neurons": int(mappings["nucleus_id"].nunique()),
                "unique_roots": int(mappings["pt_root_id"].nunique()),
                "skeleton_service_requests": 0,
                "detail": (
                    f"{len(invalid_mappings)} invalid stable-ID rows; "
                    f"{len(exact_duplicates)} exact duplicates."
                ),
            },
            {
                "lineage_step": "Stage 01 final biological candidates",
                "rows": int(len(stage01_candidates)),
                "unique_neurons": int(
                    stage01_candidates["nucleus_id"].nunique()
                ),
                "unique_roots": int(
                    stage01_candidates["pt_root_id"].nunique()
                ),
                "skeleton_service_requests": 0,
                "detail": (
                    "The 253-neuron count is persisted before Stage 02 "
                    "skeleton lookup."
                ),
            },
            {
                "lineage_step": "Stage 02 unique-root skeleton loop",
                "rows": int(len(stage02_qc)),
                "unique_neurons": int(stage02_qc["nucleus_id"].nunique()),
                "unique_roots": int(stage02_qc["pt_root_id"].nunique()),
                "skeleton_service_requests": requests,
                "detail": (
                    "One CAVE get_skeleton call per unique root: "
                    f"{int(source_counts.get('cave_skeleton_service', 0))} "
                    "new raw caches and "
                    f"{int(source_counts.get('validated_cache', 0))} "
                    "live validations of existing raw caches."
                ),
            },
        ]
    )
    if cumulative.iloc[-1]["unique_neurons"] != 253:
        raise AssertionError(
            "The cumulative audit does not reproduce the current 253 neurons."
        )
    if requests != 253:
        raise AssertionError("Stage 02 request lineage does not equal 253.")

    output_dir = root / "results/tables"
    outputs = {
        "noncumulative": (
            output_dir / "proofreading_eligibility_non_cumulative.parquet"
        ),
        "intersections": (
            output_dir / "proofreading_intersection_diagnostic.parquet"
        ),
        "cumulative": (
            output_dir / "candidate_eligibility_cumulative.parquet"
        ),
        "policy_summary": (
            output_dir / "alternative_proofreading_policy_summary.parquet"
        ),
        "policy_recordings": (
            output_dir / "alternative_proofreading_policy_recordings.parquet"
        ),
        "lineage": output_dir / "candidate_253_lineage.parquet",
    }
    tables: dict[str, pd.DataFrame] = {
        "noncumulative": noncumulative,
        "intersections": intersections,
        "cumulative": cumulative,
        "policy_summary": policy_summary,
        "policy_recordings": policy_recordings,
        "lineage": lineage,
    }
    for name, dataframe in tables.items():
        write_dataframe(dataframe, outputs[name], overwrite=True)

    if config_sha256(root) != config_hash_before:
        raise RuntimeError("project configuration changed during the audit.")
    write_stage_provenance(
        "01_candidate_eligibility_audit",
        root,
        config,
        inputs=[
            *input_paths.values(),
            stage01_candidates_path,
            stage02_qc_path,
        ],
        outputs=list(outputs.values()),
        source_metadata={
            "datastack": config["cave"]["datastack"],
            "materialization_version": int(
                config["cave"]["materialization_version"]
            ),
            "network_access": False,
            "additional_skeleton_downloads": 0,
        },
        summaries={
            "current_policy_unique_neurons": int(
                cumulative.iloc[-1]["unique_neurons"]
            ),
            "stage02_skeleton_service_requests": requests,
            "stage02_new_raw_skeletons": int(
                source_counts.get("cave_skeleton_service", 0)
            ),
            "stage02_validated_existing_caches": int(
                source_counts.get("validated_cache", 0)
            ),
        },
    )

    print("\nCUMULATIVE")
    print(cumulative[["filter", "rows", "unique_neurons"]].to_string(index=False))
    print("\nNON-CUMULATIVE")
    print(
        noncumulative[
            ["audit_group", "condition", "rows", "unique_neurons"]
        ].to_string(index=False)
    )
    print("\nPOLICY SUMMARY")
    print(policy_summary.to_string(index=False))
    print("\nLINEAGE")
    print(lineage.to_string(index=False))


if __name__ == "__main__":
    main()
