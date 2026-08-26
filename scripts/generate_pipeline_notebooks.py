"""Generate the canonical human-readable Stage 00-10 notebooks."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


STAGES = [
    ("00_source_preflight.ipynb", "Stage 00 — Source preflight", "Verify pinned CAVE sources, cached skeleton readability, software imports, and configured directories.", "microns20.cave", "run_stage00", "result[\"preflight\"]"),
    ("01_candidate_discovery.ipynb", "Stage 01 — Candidate discovery", "Normalize every manual structural-to-functional mapping and apply configured biological eligibility.", "microns20.candidates", "run_stage01", "result[\"summary\"]"),
    ("02_cave_morphology_eligibility.ipynb", "Stage 02 — CAVE morphology eligibility", "Validate canonical CAVE skeleton trees and required compartments before recording selection.", "microns20.morphology", "run_stage02", "result[\"morphology_qc\"].groupby(\"cave_morphology_eligible\").size()"),
    ("03_recording_selection.ipynb", "Stage 03 — Recording selection", "Rank feasible simultaneous recordings from CAVE structural network properties only.", "microns20.connectivity", "run_stage03", "result[\"ranking\"]"),
    ("04_connectivity_selection.ipynb", "Stage 04 — Connectivity selection", "Run the exact-size lexicographic MILP on the fully eligible selected-recording pool.", "microns20.selection", "run_stage04", "result[\"objective\"]"),
    ("05_final_qc_and_manifest.ipynb", "Stage 05 — Final QC and identity freeze", "Independently revalidate identities, mappings, CAVE skeletons, and connectivity before freezing the final manifest.", "microns20.validation", "run_stage05", "result[\"manifest\"]"),
    ("06_functional_identity.ipynb", "Stage 06 — Functional identity", "Validate all preserved CAVE manual coregistration mappings and record NDA-v8 trace acquisition as deferred.", "microns20.functional", "run_stage06", "result[\"trace_status\"]"),
    ("07_simulation_morphologies.ipynb", "Stage 07 — Simulation morphologies", "Create simulation-ready CAVE SWCs with topology-derived normalization only at nonpositive-radius points.", "microns20.morphology", "run_stage07", "result[\"morphology_qc\"]"),
    ("08_synapse_mapping.ipynb", "Stage 08 — Synapse mapping", "Map observed CAVE synapses through exact level-2/skeleton correspondence onto simulation morphology sections.", "microns20.synapses", "run_stage08", "result[\"mapping_qc\"]"),
    ("09_structural_sonata.ipynb", "Stage 09 — Structural SONATA", "Build selected and observed-external SONATA populations without inventing physiology.", "microns20.sonata", "run_stage09", "result[\"build_qc\"]"),
    ("10_end_to_end_validation.ipynb", "Stage 10 — End-to-end validation", "Independently reopen and validate identities, morphologies, node/edge populations, synapses, and readiness semantics.", "microns20.validation", "run_stage10", "result[\"validation\"]"),
]


def build_notebook(title: str, purpose: str, module: str, runner: str, display: str) -> nbf.NotebookNode:
    """Create one deterministic orchestration notebook."""

    notebook = nbf.v4.new_notebook()
    notebook.metadata["kernelspec"] = {
        "display_name": "NeuroAI",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3.10"}
    notebook.cells = [
        nbf.v4.new_markdown_cell(
            f"# {title}\n\n{purpose}\n\n"
            "Configuration is read exclusively from `configs/project.yaml`. "
            "The stage consumes the authoritative artifacts produced by its predecessor, "
            "writes its documented outputs atomically, and records hashes in `provenance/stages/`."
        ),
        nbf.v4.new_code_cell(
            "from microns20.config import find_project_root\n"
            f"from {module} import {runner}\n\n"
            "project_root = find_project_root()\n"
            f"result = {runner}(project_root)"
        ),
        nbf.v4.new_code_cell(display),
        nbf.v4.new_markdown_cell(
            "The returned values above are views of persisted artifacts; they are not a second source of truth."
        ),
    ]
    return notebook


def main() -> None:
    """Write all canonical notebooks under notebooks/pipeline."""

    project_root = Path(__file__).resolve().parents[1]
    output = project_root / "notebooks/pipeline"
    output.mkdir(parents=True, exist_ok=True)
    for filename, title, purpose, module, runner, display in STAGES:
        notebook = build_notebook(title, purpose, module, runner, display)
        nbf.write(notebook, output / filename)


if __name__ == "__main__":
    main()
