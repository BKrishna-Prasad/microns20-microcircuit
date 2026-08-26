"""CAVE structural connectivity summaries and recording selection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import networkx as nx
import pandas as pd

from microns20.artifacts import require_columns
from microns20.cave import query_synapses, validate_synapse_rows


def root_identity(candidates: pd.DataFrame) -> pd.DataFrame:
    """Return an unambiguous current-root to biological identity relation."""

    require_columns(
        candidates,
        {"nucleus_id", "pt_root_id"},
        "connectivity candidates",
    )
    identity = candidates[["nucleus_id", "pt_root_id"]].drop_duplicates()
    if identity["nucleus_id"].duplicated().any():
        raise ValueError("A nucleus maps to multiple CAVE roots.")
    if identity["pt_root_id"].duplicated().any():
        raise ValueError("A CAVE root maps to multiple nuclei.")
    if identity["pt_root_id"].le(0).any():
        raise ValueError("CAVE root IDs must be positive.")
    return identity


def collapse_synapses_to_pairs(
    synapses: pd.DataFrame,
    identity: pd.DataFrame,
) -> pd.DataFrame:
    """Count biological contacts for each directed in-pool neuron pair."""

    require_columns(
        synapses,
        {"id", "pre_pt_root_id", "post_pt_root_id"},
        "CAVE synapses",
    )
    roots = root_identity(identity)
    pre = roots.rename(
        columns={
            "nucleus_id": "source_nucleus_id",
            "pt_root_id": "pre_pt_root_id",
        }
    )
    post = roots.rename(
        columns={
            "nucleus_id": "target_nucleus_id",
            "pt_root_id": "post_pt_root_id",
        }
    )
    mapped = (
        synapses.merge(pre, on="pre_pt_root_id", how="inner", validate="many_to_one")
        .merge(post, on="post_pt_root_id", how="inner", validate="many_to_one")
    )
    if mapped["source_nucleus_id"].eq(mapped["target_nucleus_id"]).any():
        raise ValueError("Autapses remain in an autapse-free connectivity query.")
    return (
        mapped.groupby(
            ["source_nucleus_id", "target_nucleus_id"],
            as_index=False,
        )
        .agg(n_synapses=("id", "nunique"))
        .sort_values(["source_nucleus_id", "target_nucleus_id"])
        .reset_index(drop=True)
    )


def summarize_network(
    candidates: pd.DataFrame,
    synapses: pd.DataFrame,
) -> dict[str, Any]:
    """Compute configured structural recording-ranking statistics."""

    identity = root_identity(candidates)
    pairs = collapse_synapses_to_pairs(synapses, identity)
    node_ids = sorted(identity["nucleus_id"].astype(int).tolist())
    graph = nx.DiGraph()
    graph.add_nodes_from(node_ids)
    graph.add_edges_from(
        pairs[["source_nucleus_id", "target_nucleus_id"]]
        .astype(int)
        .itertuples(index=False, name=None)
    )
    n_nodes = len(node_ids)
    possible = n_nodes * (n_nodes - 1)
    weak_components = list(nx.weakly_connected_components(graph))
    return {
        "n_candidates": int(n_nodes),
        "n_synaptic_contacts": int(synapses["id"].nunique()) if not synapses.empty else 0,
        "n_connected_directed_pairs": int(len(pairs)),
        "density": (
            float(len(pairs) / possible) if possible > 0 else 0.0
        ),
        "largest_weak_component": int(
            max((len(component) for component in weak_components), default=0)
        ),
    }


def recording_networks(
    client: Any,
    candidates: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Query and summarize CAVE recurrence for every feasible recording."""

    require_columns(
        candidates,
        {"nucleus_id", "pt_root_id", "session", "scan_idx"},
        "recording candidates",
    )
    summaries = []
    synapse_tables = []
    for (session, scan_idx), recording in candidates.groupby(
        ["session", "scan_idx"],
        sort=True,
    ):
        roots = recording["pt_root_id"].astype(int).unique().tolist()
        synapses = query_synapses(
            client,
            config,
            pre_ids=roots,
            post_ids=roots,
        )
        if synapses.empty:
            synapses = pd.DataFrame(
                columns=[
                    "id",
                    "pre_pt_supervoxel_id",
                    "pre_pt_root_id",
                    "post_pt_supervoxel_id",
                    "post_pt_root_id",
                ]
            )
        else:
            synapses = validate_synapse_rows(synapses)
        summaries.append(
            {
                "session": int(session),
                "scan_idx": int(scan_idx),
                **summarize_network(recording, synapses),
            }
        )
        if not synapses.empty:
            tagged = synapses.copy()
            tagged.insert(0, "scan_idx", int(scan_idx))
            tagged.insert(0, "session", int(session))
            synapse_tables.append(tagged)

    all_synapses = (
        pd.concat(synapse_tables, ignore_index=True)
        if synapse_tables
        else pd.DataFrame()
    )
    return pd.DataFrame(summaries), all_synapses


def rank_recordings(
    summary: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Rank structurally feasible recordings with deterministic tie-breaking."""

    policy = config["recording_selection"]
    ranking = list(policy["ranking"])
    ties = list(policy["deterministic_tie_breakers"])
    require_columns(
        summary,
        {"n_candidates", *ranking, *ties},
        "recording summary",
    )
    feasible = summary.loc[
        summary["n_candidates"].ge(int(policy["minimum_candidates"]))
    ].copy()
    if feasible.empty:
        raise RuntimeError(
            "No recording has the configured minimum number of biologically and "
            "CAVE-morphology eligible candidates."
        )
    descending = bool(policy["descending"])
    ranked = feasible.sort_values(
        ranking + ties,
        ascending=[not descending] * len(ranking) + [True] * len(ties),
        kind="mergesort",
    ).reset_index(drop=True)
    ranked.insert(0, "recording_rank", range(1, len(ranked) + 1))
    ranked["selected_recording"] = False
    ranked.loc[0, "selected_recording"] = True
    return ranked


def selected_recording_key(ranked: pd.DataFrame) -> tuple[int, int]:
    """Return the unique selected session/scan from a ranked table."""

    selected = ranked.loc[ranked["selected_recording"]]
    if len(selected) != 1:
        raise ValueError("Recording ranking must select exactly one recording.")
    row = selected.iloc[0]
    return int(row["session"]), int(row["scan_idx"])


def filter_selected_recording(
    candidates: pd.DataFrame,
    synapses: pd.DataFrame,
    roi_mappings: pd.DataFrame,
    ranked: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create authoritative selected-recording artifacts."""

    session, scan_idx = selected_recording_key(ranked)
    candidate_rows = candidates.loc[
        candidates["session"].eq(session)
        & candidates["scan_idx"].eq(scan_idx)
    ].copy()
    synapse_rows = synapses.loc[
        synapses["session"].eq(session)
        & synapses["scan_idx"].eq(scan_idx)
    ].drop(columns=["session", "scan_idx"]).copy()
    mapping_rows = roi_mappings.loc[
        roi_mappings["session"].eq(session)
        & roi_mappings["scan_idx"].eq(scan_idx)
    ].copy()

    expected = int(candidate_rows["n_functional_mappings"].sum())
    if len(mapping_rows) != expected:
        raise ValueError(
            f"Selected recording lost functional mappings: "
            f"expected {expected}, found {len(mapping_rows)}."
        )
    return (
        candidate_rows.sort_values("nucleus_id").reset_index(drop=True),
        validate_synapse_rows(synapse_rows),
        mapping_rows.sort_values(
            ["nucleus_id", "unit_id", "field"]
        ).reset_index(drop=True),
    )


def compare_synapse_snapshots(
    expected: pd.DataFrame,
    observed: pd.DataFrame,
) -> pd.DataFrame:
    """Independently compare authoritative synapse IDs and endpoints."""

    columns = [
        "id",
        "pre_pt_root_id",
        "post_pt_root_id",
        "pre_pt_supervoxel_id",
        "post_pt_supervoxel_id",
    ]
    require_columns(expected, columns, "expected synapses")
    require_columns(observed, columns, "observed synapses")
    left = expected[columns].rename(
        columns={column: f"{column}_expected" for column in columns[1:]}
    )
    right = observed[columns].rename(
        columns={column: f"{column}_observed" for column in columns[1:]}
    )
    compared = left.merge(right, on="id", how="outer", indicator=True)
    compared["endpoints_agree"] = compared["_merge"].eq("both")
    for column in columns[1:]:
        compared["endpoints_agree"] &= compared[
            f"{column}_expected"
        ].eq(compared[f"{column}_observed"])
    return compared.sort_values("id").reset_index(drop=True)


def run_stage03(project_root: str | None = None) -> dict[str, Any]:
    """Rank feasible simultaneous recordings using CAVE structure only."""

    from microns20 import candidates as candidate_logic, cave
    from microns20.artifacts import write_dataframe
    from microns20.orchestration import artifact_path, project_context
    from microns20.provenance import require_completed_stage, write_stage_provenance
    from microns20.qc import population_summary

    root, config = project_context(project_root)
    require_completed_stage("02_cave_morphology_eligibility", root, config)
    candidate_path = artifact_path(root, config, "candidates", "morphology_eligible_candidates.parquet")
    all_mapping_path = artifact_path(root, config, "candidates", "functional_mappings.parquet")
    candidates = pd.read_parquet(candidate_path)
    all_mappings = pd.read_parquet(all_mapping_path)
    mappings = candidate_logic.functional_mappings_for_candidates(all_mappings, candidates)
    counts = candidates.groupby(["session", "scan_idx"], as_index=False).agg(n_candidates=("nucleus_id", "nunique")).sort_values(["session", "scan_idx"])
    minimum = int(config["recording_selection"]["minimum_candidates"])
    feasible_keys = counts.loc[counts["n_candidates"].ge(minimum), ["session", "scan_idx"]]
    feasible = candidates.merge(feasible_keys, on=["session", "scan_idx"], how="inner", validate="many_to_one")
    if feasible.empty:
        raise RuntimeError("No recording has the configured minimum candidate count.")
    client = cave.create_client(config)
    network_summary, all_synapses = recording_networks(client, feasible, config)
    ranking = rank_recordings(network_summary, config)
    selected_candidates, selected_synapses, selected_mappings = filter_selected_recording(feasible, all_synapses, mappings, ranking)
    summary_output = artifact_path(root, config, "results_tables", "recording_selection_summary.parquet")
    candidate_output = artifact_path(root, config, "recordings", "selected_recording_candidates.parquet")
    synapse_output = artifact_path(root, config, "recordings", "selected_recording_synapses.parquet")
    mapping_output = artifact_path(root, config, "recordings", "selected_recording_functional_mappings.parquet")
    for dataframe, path in [(ranking, summary_output), (selected_candidates, candidate_output), (selected_synapses, synapse_output), (selected_mappings, mapping_output)]:
        write_dataframe(dataframe, path, overwrite=True)
    session, scan_idx = selected_recording_key(ranking)
    provenance = write_stage_provenance(
        "03_recording_selection", root, config,
        inputs=[candidate_path, all_mapping_path],
        outputs=[summary_output, candidate_output, synapse_output, mapping_output],
        source_metadata=cave.source_metadata(client, config),
        summaries={
            "selection_source": "CAVE structural connectivity",
            "functional_trace_gate": False,
            "selected_session": session,
            "selected_scan_idx": scan_idx,
            "selected_recording": population_summary(selected_candidates),
            "recording_metrics": ranking.to_dict("records"),
        },
    )
    return {"ranking": ranking, "selected_candidates": selected_candidates, "selected_synapses": selected_synapses, "selected_mappings": selected_mappings, "provenance": provenance}
