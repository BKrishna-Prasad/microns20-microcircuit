"""End-to-end tests for the CAVE-first structural pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from bmtk.builder.bionet import SWCReader
import h5py
import morphio
import nbformat
import numpy as np
import pandas as pd
import pytest

from microns20.config import load_config
from microns20.morphology import read_swc
from microns20.validation import (
    validate_frozen_manifest,
    validate_structural_sonata,
)


@pytest.fixture(scope="session")
def root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def config(root: Path) -> dict:
    return load_config(root)


@pytest.fixture(scope="session")
def manifest(root: Path) -> pd.DataFrame:
    return pd.read_parquet(root / "data/processed/final20_manifest.parquet")


@pytest.fixture(scope="session")
def mappings(root: Path) -> pd.DataFrame:
    return pd.read_parquet(root / "data/processed/final20_functional_mappings.parquet")


def test_selected_identity_and_model_ids(config: dict, manifest: pd.DataFrame, mappings: pd.DataFrame) -> None:
    validate_frozen_manifest(manifest, mappings, config)
    assert len(manifest) == config["selection"]["n_neurons"]
    expected = list(range(config["selection"]["model_id_start"], config["selection"]["model_id_start"] + len(manifest)))
    assert manifest.sort_values("nucleus_id")["model_node_id"].tolist() == expected
    for column in ("nucleus_id", "pt_root_id", "pt_supervoxel_id"):
        assert manifest[column].gt(0).all()
        assert manifest[column].is_unique


def test_candidate_lineage_and_morphology_gate(root: Path) -> None:
    biological = pd.read_parquet(root / "data/interim/candidates/biological_candidates.parquet")
    eligible = pd.read_parquet(root / "data/interim/candidates/morphology_eligible_candidates.parquet")
    qc = pd.read_parquet(root / "results/tables/cave_morphology_qc.parquet")
    assert "model_node_id" not in biological.columns
    assert biological["biologically_eligible"].all()
    assert eligible["cave_morphology_eligible"].all()
    required = qc["n_roots"].eq(1) & qc["n_soma_points"].gt(0) & qc["n_axon_points"].gt(0) & qc["n_dendrite_points"].gt(0)
    assert qc["cave_morphology_eligible"].equals(required & qc["finite_coordinates"] & qc["finite_radii"] & qc["unique_point_ids"] & qc["valid_parent_references"] & qc["acyclic"] & qc["connected"] & qc["allowed_types"] & qc["morphio_loadable"] & qc["bmtk_neuron_loadable"])
    assert qc.loc[qc["requires_radius_normalization"], "n_nonpositive_radius_points"].gt(0).all()


def test_recording_is_structurally_derived(root: Path, manifest: pd.DataFrame) -> None:
    ranking = pd.read_parquet(root / "results/tables/recording_selection_summary.parquet")
    selected = ranking.loc[ranking["selected_recording"]].iloc[0]
    assert int(selected["recording_rank"]) == 1
    assert manifest["session"].eq(int(selected["session"])).all()
    assert manifest["scan_idx"].eq(int(selected["scan_idx"])).all()
    assert "functional_correlation" not in " ".join(ranking.columns).lower()


def test_functional_mapping_multiplicity_is_preserved(root: Path, manifest: pd.DataFrame, mappings: pd.DataFrame) -> None:
    counts = mappings.groupby("model_node_id").size()
    expected = manifest.set_index("model_node_id")["n_functional_mappings"]
    assert counts.sort_index().equals(expected.astype("int64").sort_index())
    keys = ["model_node_id", "nucleus_id", "session", "scan_idx", "unit_id", "field"]
    assert not mappings.duplicated(keys).any()
    status = json.loads((root / "data/processed/functional/functional_trace_acquisition_status.json").read_text())
    assert status["status"] == "DEFERRED_POST_SONATA"
    assert status["empty_trace_artifacts_written"] is False


def test_radius_normalization_preserves_cave_geometry(root: Path) -> None:
    morphology_manifest = pd.read_parquet(root / "data/processed/morphologies/manifest.parquet")
    point_map = pd.read_parquet(root / "data/interim/morphologies/cave_to_simulation_point_map.parquet")
    for row in morphology_manifest.itertuples(index=False):
        raw = read_swc(root / row.source_cave_skeleton).rename(columns={"id": "raw_cave_point_id"})
        simulation = read_swc(root / row.simulation_morphology).rename(columns={"id": "simulation_point_id"})
        mapping = point_map.loc[point_map["model_node_id"].eq(row.model_node_id)].rename(columns={"raw_point_id": "raw_cave_point_id"})
        paired = mapping.merge(raw, on="raw_cave_point_id").merge(simulation, on="simulation_point_id", suffixes=("_raw", "_simulation"))
        for column in ("x", "y", "z", "type"):
            assert np.array_equal(paired[f"{column}_raw"], paired[f"{column}_simulation"])
        positive = paired["radius_raw"].gt(0)
        assert np.array_equal(paired.loc[positive, "radius_raw"], paired.loc[positive, "radius_simulation"])
        assert paired["radius_simulation"].gt(0).all()


def test_all_simulation_morphologies_reload(root: Path, config: dict) -> None:
    morphology_manifest = pd.read_parquet(root / "data/processed/morphologies/manifest.parquet")
    qc = pd.read_parquet(root / "results/tables/simulation_morphology_qc.parquet")
    assert len(morphology_manifest) == config["selection"]["n_neurons"]
    assert qc["simulation_morphology_valid"].all()
    for relative in morphology_manifest["simulation_morphology"]:
        path = root / relative
        morphio.Morphology(str(path))
        assert len(SWCReader(str(path)).sections) > 0


def test_synapses_are_one_to_one_and_finite(root: Path, manifest: pd.DataFrame) -> None:
    selected_roots = set(manifest["pt_root_id"].astype(int))
    intrinsic = pd.read_parquet(root / "data/processed/connectivity/intrinsic_synapses.parquet")
    external = pd.read_parquet(root / "data/processed/connectivity/incoming_external_synapses.parquet")
    unresolved = pd.read_parquet(root / "data/processed/connectivity/unresolved_synapses.parquet")
    assert unresolved.empty
    assert intrinsic["id"].is_unique and external["id"].is_unique
    assert set(intrinsic["pre_pt_root_id"].astype(int)).issubset(selected_roots)
    assert set(intrinsic["post_pt_root_id"].astype(int)).issubset(selected_roots)
    assert not set(external["pre_pt_root_id"].astype(int)) & selected_roots
    for frame in (intrinsic, external):
        assert frame["post_pt_root_id"].astype(int).isin(selected_roots).all()
        assert frame["afferent_section_pos"].between(0, 1).all()
        assert np.isfinite(frame[["afferent_center_x_um", "afferent_center_y_um", "afferent_center_z_um", "synapse_to_section_distance_um"]].to_numpy()).all()


def test_structural_sonata_independent_validation(root: Path, config: dict, manifest: pd.DataFrame, mappings: pd.DataFrame) -> None:
    result = validate_structural_sonata(
        manifest,
        mappings,
        pd.read_parquet(root / "data/processed/morphologies/manifest.parquet"),
        pd.read_parquet(root / "data/processed/connectivity/external_presynaptic_nodes.parquet"),
        pd.read_parquet(root / "data/processed/connectivity/intrinsic_synapses.parquet"),
        pd.read_parquet(root / "data/processed/connectivity/incoming_external_synapses.parquet"),
        root,
        root / "data/processed/sonata",
        config,
    )
    assert result["structural_sonata_valid"] is True
    assert result["ready_for_emodel_attachment"] is True
    assert result["ready_for_functional_validation"] is False
    assert result["ready_for_biophysical_simulation"] is False
    assert result["physiology_assigned"] is False


def test_sonata_edge_ids_equal_processed_synapses(root: Path) -> None:
    pairs = [
        ("microns20_to_microns20", "intrinsic_synapses.parquet"),
        ("external_observed_to_microns20", "incoming_external_synapses.parquet"),
    ]
    for population_name, parquet_name in pairs:
        path = root / "data/processed/sonata/network/edges" / f"{population_name}_edges.h5"
        expected = pd.read_parquet(root / "data/processed/connectivity" / parquet_name)["id"].to_numpy()
        with h5py.File(path, "r") as handle:
            population = handle[f"edges/{population_name}"]
            index = np.asarray(population["edge_group_index"], dtype=np.int64)
            observed = np.asarray(population["0/synapse_id"], dtype=np.int64)[index]
        assert np.array_equal(np.sort(observed), np.sort(expected))


def test_provenance_complete_and_pipeline_notebooks_valid(root: Path) -> None:
    stages = [
        "00_source_preflight",
        "01_candidate_discovery",
        "02_cave_morphology_eligibility",
        "03_recording_selection",
        "04_cave_connectivity_selection",
        "05_final_qc_and_manifest",
        "06_functional_activity",
        "07_simulation_morphologies",
        "08_synapse_mapping",
        "09_structural_sonata",
        "10_end_to_end_validation",
    ]

    for stage in stages:
        record_path = root / "provenance/stages" / f"{stage}.json"
        assert record_path.is_file(), record_path

        record = json.loads(record_path.read_text())
        assert record["status"] == "complete"
        assert record["config_sha256"]

    names = [
        "00_source_preflight.ipynb",
        "01_candidate_discovery.ipynb",
        "02_cave_morphology_eligibility.ipynb",
        "03_recording_selection.ipynb",
        "04_connectivity_selection.ipynb",
        "05_final_qc_and_manifest.ipynb",
        "06_functional_identity.ipynb",
        "07_simulation_morphologies.ipynb",
        "08_synapse_mapping.ipynb",
        "09_structural_sonata.ipynb",
        "10_end_to_end_validation.ipynb",
    ]

    pipeline_notebooks = [
        root / "notebooks/pipeline" / name
        for name in names
    ]

    for path in pipeline_notebooks:
        assert path.is_file(), path

        notebook = nbformat.read(path, as_version=4)

        errors = [
            output
            for cell in notebook.cells
            if cell.cell_type == "code"
            for output in cell.get("outputs", [])
            if output.output_type == "error"
        ]

        assert not errors, path.name

def test_selected_ids_are_not_hardcoded_in_source(root: Path, manifest: pd.DataFrame) -> None:
    source_paths = [path for path in (root / "src/microns20").glob("*.py") if path.name not in {"pipeline.py"}]
    source_text = "\n".join(path.read_text() for path in source_paths)
    for column in ("nucleus_id", "pt_root_id", "pt_supervoxel_id"):
        for value in manifest[column].astype(str):
            assert value not in source_text
    config_text = (root / "configs/project.yaml").read_text().lower()
    assert "selected_session" not in config_text
    assert "selected_scan" not in config_text
    assert "obi" not in config_text
