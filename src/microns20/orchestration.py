"""Shared file-level orchestration helpers for canonical stages."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
import shutil

from microns20.artifacts import sha256_file
from microns20.config import configured_path, find_project_root, load_config


def project_context(project_root: str | Path | None = None) -> tuple[Path, dict[str, Any]]:
    """Resolve the project root and load the sole canonical configuration."""

    root = find_project_root(project_root)
    return root, load_config(root)


def artifact_path(
    root: Path,
    config: Mapping[str, Any],
    directory_key: str,
    filename: str,
) -> Path:
    """Resolve a configured artifact directory plus a filename."""

    return configured_path(root, config, directory_key) / filename


def archive_replaced_artifact(path: Path, root: Path) -> Path | None:
    """Content-address an artifact before an intentional canonical replacement."""

    if not path.is_file():
        return None
    digest = sha256_file(path)
    target = (
        root
        / "provenance/legacy/pre_cave_first_freeze"
        / f"{path.stem}_{digest[:16]}{path.suffix}"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256_file(target) != digest:
            raise RuntimeError(f"Legacy archive hash conflict: {target}")
    else:
        shutil.copy2(path, target)
    return target
