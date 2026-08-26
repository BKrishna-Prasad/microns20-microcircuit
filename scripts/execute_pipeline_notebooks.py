"""Execute pipeline notebooks sequentially in the current environment."""

from __future__ import annotations

from pathlib import Path

from nbclient import NotebookClient
import nbformat


NOTEBOOKS = [
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


def main() -> None:
    """Execute and persist every notebook, stopping at the first failure."""

    root = Path(__file__).resolve().parents[1]
    for name in NOTEBOOKS:
        path = root / "notebooks/pipeline" / name
        print(f"Running {name}", flush=True)
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(
            notebook,
            timeout=3600,
            kernel_name="python3",
            resources={"metadata": {"path": str(root)}},
        )
        client.execute()
        nbformat.write(notebook, path)
        print(f"Completed {name}", flush=True)


if __name__ == "__main__":
    main()
