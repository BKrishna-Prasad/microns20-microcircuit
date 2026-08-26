"""MILP selection of the structurally densest exact-size CAVE subnetwork."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from microns20.artifacts import require_columns
from microns20.connectivity import collapse_synapses_to_pairs


PAIR_COLUMNS = [
    "source_nucleus_id",
    "target_nucleus_id",
    "n_synapses",
]


def validate_pair_table(
    pair_table: pd.DataFrame,
    candidate_ids: Iterable[int],
) -> pd.DataFrame:
    """Validate unique positive directed pair counts inside the candidate set."""

    require_columns(pair_table, PAIR_COLUMNS, "connectivity pairs")
    nodes = {int(value) for value in candidate_ids}
    pairs = pair_table.loc[
        pair_table["source_nucleus_id"].isin(nodes)
        & pair_table["target_nucleus_id"].isin(nodes)
    ].copy()
    if pairs.duplicated(
        ["source_nucleus_id", "target_nucleus_id"]
    ).any():
        raise ValueError("Directed connectivity pairs are not unique.")
    if pairs["source_nucleus_id"].eq(pairs["target_nucleus_id"]).any():
        raise ValueError("Autaptic pairs are not eligible for the MILP.")
    if pairs["n_synapses"].le(0).any():
        raise ValueError("Connected pairs must have positive synapse counts.")
    return pairs.sort_values(PAIR_COLUMNS[:2]).reset_index(drop=True)


def select_densest_subnetwork(
    candidate_nucleus_ids: Iterable[int],
    pair_table: pd.DataFrame,
    n_select: int,
) -> np.ndarray:
    """Select exactly n neurons with lexicographic pair/contact objectives."""

    node_ids = np.asarray(
        sorted({int(value) for value in candidate_nucleus_ids}),
        dtype=np.int64,
    )
    if n_select <= 0:
        raise ValueError("n_select must be positive.")
    if n_select > len(node_ids):
        raise ValueError(
            f"Cannot select {n_select} neurons from {len(node_ids)} candidates."
        )
    pairs = validate_pair_table(pair_table, node_ids)
    node_index = {
        int(node_id): index for index, node_id in enumerate(node_ids)
    }
    n_nodes = len(node_ids)
    n_pairs = len(pairs)
    n_variables = n_nodes + n_pairs

    max_contact_difference = int(pairs["n_synapses"].sum())
    primary_weight = max_contact_difference + 1
    objective = np.zeros(n_variables, dtype=float)
    for edge_index, row in pairs.iterrows():
        objective[n_nodes + edge_index] = -(
            primary_weight + int(row["n_synapses"])
        )

    rows = []
    lower = []
    upper = []
    exact_size = np.zeros(n_variables)
    exact_size[:n_nodes] = 1
    rows.append(exact_size)
    lower.append(float(n_select))
    upper.append(float(n_select))

    for edge_index, edge in pairs.iterrows():
        source = node_index[int(edge["source_nucleus_id"])]
        target = node_index[int(edge["target_nucleus_id"])]
        selected_pair = n_nodes + edge_index

        row = np.zeros(n_variables)
        row[selected_pair] = 1
        row[source] = -1
        rows.append(row)
        lower.append(-np.inf)
        upper.append(0.0)

        row = np.zeros(n_variables)
        row[selected_pair] = 1
        row[target] = -1
        rows.append(row)
        lower.append(-np.inf)
        upper.append(0.0)

        row = np.zeros(n_variables)
        row[source] = 1
        row[target] = 1
        row[selected_pair] = -1
        rows.append(row)
        lower.append(-np.inf)
        upper.append(1.0)

    constraints = LinearConstraint(
        np.vstack(rows),
        np.asarray(lower),
        np.asarray(upper),
    )
    result = milp(
        c=objective,
        integrality=np.ones(n_variables),
        bounds=Bounds(
            np.zeros(n_variables),
            np.ones(n_variables),
        ),
        constraints=constraints,
        options={"presolve": True},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Subnetwork optimization failed: {result.message}")

    selected = node_ids[result.x[:n_nodes] > 0.5]
    if len(selected) != n_select:
        raise AssertionError(
            f"MILP returned {len(selected)} neurons, expected {n_select}."
        )
    return selected


def select_final_population(
    candidates: pd.DataFrame,
    synapses: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Run the configured MILP and return an unfrozen biological population."""

    require_columns(
        candidates,
        {"nucleus_id", "pt_root_id"},
        "selected-recording candidates",
    )
    if candidates["nucleus_id"].duplicated().any():
        raise ValueError("MILP candidates must have one row per biological neuron.")

    pairs = collapse_synapses_to_pairs(synapses, candidates)
    n_select = int(config["selection"]["n_neurons"])
    selected_ids = select_densest_subnetwork(
        candidates["nucleus_id"].astype(int).tolist(),
        pairs,
        n_select,
    )
    selected = candidates.loc[
        candidates["nucleus_id"].isin(selected_ids)
    ].copy()
    if "model_node_id" in selected.columns:
        raise ValueError("model_node_id must not exist before Stage 05.")

    selected_pairs = pairs.loc[
        pairs["source_nucleus_id"].isin(selected_ids)
        & pairs["target_nucleus_id"].isin(selected_ids)
    ].copy()
    objective = {
        "n_selected_neurons": int(len(selected)),
        "n_connected_pairs": int(len(selected_pairs)),
        "n_synapses": int(selected_pairs["n_synapses"].sum()),
    }
    return (
        selected.sort_values("nucleus_id").reset_index(drop=True),
        selected_pairs.reset_index(drop=True),
        objective,
    )


def run_stage04(project_root: str | None = None) -> dict[str, Any]:
    """Run the exact-size lexicographic CAVE connectivity MILP."""

    from microns20.artifacts import write_dataframe
    from microns20.orchestration import artifact_path, project_context
    from microns20.provenance import require_completed_stage, write_stage_provenance

    root, config = project_context(project_root)
    require_completed_stage("03_recording_selection", root, config)
    candidate_path = artifact_path(root, config, "recordings", "selected_recording_candidates.parquet")
    synapse_path = artifact_path(root, config, "recordings", "selected_recording_synapses.parquet")
    candidates = pd.read_parquet(candidate_path)
    synapses = pd.read_parquet(synapse_path)
    selected, selected_pairs, objective = select_final_population(candidates, synapses, config)
    output = artifact_path(root, config, "selection", "selected_population_unfrozen.parquet")
    connectivity_output = artifact_path(root, config, "results_tables", "final20_selection_connectivity.parquet")
    write_dataframe(selected, output, overwrite=True)
    write_dataframe(selected_pairs, connectivity_output, overwrite=True)
    provenance = write_stage_provenance(
        "04_cave_connectivity_selection", root, config,
        inputs=[candidate_path, synapse_path], outputs=[output, connectivity_output],
        summaries=objective,
    )
    return {"selected_population": selected, "selected_pairs": selected_pairs, "objective": objective, "provenance": provenance}
