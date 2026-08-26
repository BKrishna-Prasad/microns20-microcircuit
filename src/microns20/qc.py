"""Descriptive CAVE-first source, spatial, and population QC."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from microns20.artifacts import assert_directory_writable, require_columns
from microns20.config import configured_path


def path_preflight(
    project_root: str | Path,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Validate every configured directory is contained and writable."""

    rows = []
    for key in config["paths"]:
        path = configured_path(project_root, config, str(key))
        assert_directory_writable(path)
        rows.append(
            {"check": f"path:{key}", "status": "pass", "detail": str(path)}
        )
    return pd.DataFrame(rows)


def soma_spatial_qc(
    neurons: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Calculate data-derived soma spatial metrics from CAVE coordinates."""

    coordinate_options = [
        ("pt_position_x", "pt_position_y", "pt_position_z"),
        ("pt_x", "pt_y", "pt_z"),
        ("soma_x_um", "soma_y_um", "soma_z_um"),
    ]
    coordinate_columns = next(
        (
            columns
            for columns in coordinate_options
            if set(columns).issubset(neurons.columns)
        ),
        None,
    )
    if coordinate_columns is None:
        raise ValueError("No CAVE soma-coordinate triplet is available.")
    require_columns(neurons, {"nucleus_id", *coordinate_columns}, "neurons")
    positions = neurons[list(coordinate_columns)].to_numpy(dtype=float)
    if len(positions) < 2 or not np.isfinite(positions).all():
        raise ValueError("Spatial QC requires at least two finite somata.")

    centroid = positions.mean(axis=0)
    spans = positions.max(axis=0) - positions.min(axis=0)
    distances = pdist(positions)
    from_centroid = np.linalg.norm(positions - centroid, axis=1)
    identity_columns = [
        column
        for column in ["model_node_id", "nucleus_id", "pt_root_id"]
        if column in neurons.columns
    ]
    per_neuron = neurons[identity_columns].copy()
    per_neuron["distance_from_centroid_um"] = from_centroid
    summary = {
        "coordinate_columns": list(coordinate_columns),
        "centroid_um": centroid.tolist(),
        "span_um": spans.tolist(),
        "median_pairwise_distance_um": float(np.median(distances)),
        "mean_pairwise_distance_um": float(np.mean(distances)),
        "maximum_pairwise_distance_um": float(np.max(distances)),
        "maximum_distance_from_centroid_um": float(np.max(from_centroid)),
    }
    return per_neuron, summary


def population_summary(population: pd.DataFrame) -> dict[str, Any]:
    """Summarize identities and recording membership from actual rows."""

    require_columns(
        population,
        {"nucleus_id", "pt_root_id", "session", "scan_idx"},
        "population",
    )
    recordings = (
        population[["session", "scan_idx"]]
        .drop_duplicates()
        .sort_values(["session", "scan_idx"])
        .to_dict("records")
    )
    return {
        "n_rows": int(len(population)),
        "n_neurons": int(population["nucleus_id"].nunique()),
        "n_roots": int(population["pt_root_id"].nunique()),
        "recordings": recordings,
    }
