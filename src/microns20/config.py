"""Canonical configuration loading and project-relative path helpers."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from pathlib import Path
from typing import Any

import yaml


CONFIG_RELATIVE_PATH = Path("configs/project.yaml")


def find_project_root(start: str | Path | None = None) -> Path:
    """Find the nearest parent containing the canonical configuration."""

    current = Path(start or Path.cwd()).expanduser().resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / CONFIG_RELATIVE_PATH).is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find {CONFIG_RELATIVE_PATH} from {current} or its parents."
    )


def canonical_config_path(project_root: str | Path | None = None) -> Path:
    """Return the only supported project configuration path."""

    root = find_project_root(project_root)
    return root / CONFIG_RELATIVE_PATH


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}.")
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate intentional policies required by the CAVE-first pipeline."""

    required_sections = {
        "project",
        "cave",
        "functional",
        "eligibility",
        "recording_selection",
        "selection",
        "morphology",
        "synapses",
        "sonata",
        "paths",
    }
    missing = required_sections - set(config)
    if missing:
        raise ValueError(f"Configuration is missing sections: {sorted(missing)}")

    cave = _require_mapping(config["cave"], "cave")
    for key in (
        "datastack",
        "materialization_version",
        "skeleton_version",
        "desired_resolution_um",
        "tables",
    ):
        if key not in cave:
            raise ValueError(f"cave.{key} is required.")
    if int(cave["materialization_version"]) <= 0:
        raise ValueError("cave.materialization_version must be positive.")
    if int(cave["skeleton_version"]) <= 0:
        raise ValueError("cave.skeleton_version must be positive.")
    resolution = list(cave["desired_resolution_um"])
    if len(resolution) != 3 or any(float(value) <= 0 for value in resolution):
        raise ValueError("cave.desired_resolution_um must have three positives.")

    tables = _require_mapping(cave["tables"], "cave.tables")
    required_tables = {
        "manual_coregistration",
        "functional_area",
        "proofreading",
        "cell_types",
        "cell_type_corrections",
        "mtypes",
        "mtype_corrections",
        "synapses",
    }
    missing_tables = required_tables - set(tables)
    if missing_tables:
        raise ValueError(
            f"cave.tables is missing configured sources: {sorted(missing_tables)}"
        )

    functional = _require_mapping(config["functional"], "functional")
    if functional.get("identity_source") != "cave_manual_coregistration":
        raise ValueError("Functional identity must come from CAVE manual coregistration.")
    if bool(functional.get("require_trace_for_selection")):
        raise ValueError("Functional traces must not gate structural selection.")
    if functional.get("trace_acquisition_status") != "deferred_post_sonata":
        raise ValueError("Functional trace acquisition must be explicitly deferred.")
    if not functional.get("planned_trace_source"):
        raise ValueError("functional.planned_trace_source is required.")
    if not bool(functional.get("preserve_all_manual_mappings")):
        raise ValueError("Every manual functional mapping must be preserved.")

    skeleton = _require_mapping(
        _require_mapping(config["eligibility"], "eligibility")["skeleton"],
        "eligibility.skeleton",
    )
    allowed_types = {int(value) for value in skeleton["allowed_types"]}
    if allowed_types != {1, 2, 3, 4}:
        raise ValueError("CAVE morphology types must be exactly {1, 2, 3, 4}.")
    if not bool(skeleton.get("require_finite_radii")):
        raise ValueError("Finite radii remain a hard skeleton requirement.")
    if bool(skeleton.get("require_positive_radii")):
        raise ValueError(
            "Nonpositive finite radii are normalized only after population freeze."
        )
    if bool(skeleton.get("require_apical")):
        raise ValueError("Apical annotation is not a hard eligibility requirement.")

    selection = _require_mapping(config["selection"], "selection")
    if int(selection["n_neurons"]) <= 0:
        raise ValueError("selection.n_neurons must be positive.")
    if list(selection["model_id_sort_keys"]) != ["nucleus_id"]:
        raise ValueError("Model IDs must be deterministic by nucleus_id.")

    paths = _require_mapping(config["paths"], "paths")
    for key, value in paths.items():
        path = Path(str(value))
        if path.is_absolute():
            raise ValueError(f"paths.{key} must be project-relative, got {path}.")


def load_config(project_root: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the canonical project configuration only."""

    path = canonical_config_path(project_root)
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    config = dict(_require_mapping(loaded, "project configuration"))
    validate_config(config)
    return config


def project_path(
    project_root: str | Path,
    relative_path: str | Path,
) -> Path:
    """Resolve a configured path and require that it remains in the project."""

    root = Path(project_root).expanduser().resolve()
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError(f"Expected a project-relative path, got {relative}.")
    resolved = (root / relative).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Path escapes project root: {relative}")
    return resolved


def configured_path(
    project_root: str | Path,
    config: Mapping[str, Any],
    key: str,
) -> Path:
    """Resolve one entry from the canonical paths mapping."""

    paths = _require_mapping(config["paths"], "paths")
    if key not in paths:
        raise KeyError(f"Unknown configured path key: {key}")
    return project_path(project_root, paths[key])


def cave_raw_directory(
    project_root: str | Path,
    config: Mapping[str, Any],
) -> Path:
    """Return the version-pinned raw CAVE directory."""

    cave = config["cave"]
    return (
        configured_path(project_root, config, "raw_cave")
        / str(cave["datastack"])
        / f"materialization_{int(cave['materialization_version'])}"
    )


def dandi_raw_directory(
    project_root: str | Path,
    config: Mapping[str, Any],
) -> Path:
    """Return the version-pinned DANDI metadata directory."""

    dandi = config["functional"]["dandi"]
    return (
        configured_path(project_root, config, "raw_dandi")
        / str(dandi["dandiset_id"])
        / str(dandi["version"])
    )


def config_sha256(project_root: str | Path | None = None) -> str:
    """Return the SHA-256 of the canonical configuration bytes."""

    path = canonical_config_path(project_root)
    return hashlib.sha256(path.read_bytes()).hexdigest()
