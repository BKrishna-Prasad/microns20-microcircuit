"""Stage provenance and immutable source-manifest helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
import json
import platform
import sys
from typing import Any

import pandas as pd

from microns20.artifacts import artifact_record, sha256_file, write_json
from microns20.config import config_sha256, configured_path


def utc_now() -> str:
    """Return a timezone-aware UTC timestamp for provenance."""

    return datetime.now(timezone.utc).isoformat()


def software_environment(package_names: Sequence[str]) -> dict[str, Any]:
    """Record Python and installed package versions without changing the env."""

    from importlib.metadata import PackageNotFoundError, version

    packages: dict[str, str | None] = {}
    for name in package_names:
        try:
            packages[str(name)] = version(str(name))
        except PackageNotFoundError:
            packages[str(name)] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
    }


def artifact_records(
    paths: Sequence[str | Path],
    project_root: str | Path,
) -> list[dict[str, Any]]:
    """Describe existing files, including dataframe schema when applicable."""

    records = []
    for path in paths:
        target = Path(path)
        dataframe = None
        if target.suffix in {".parquet", ".csv"}:
            dataframe = (
                pd.read_parquet(target)
                if target.suffix == ".parquet"
                else pd.read_csv(target)
            )
        records.append(
            artifact_record(target, project_root, dataframe=dataframe)
        )
    return records


def write_stage_provenance(
    stage: str,
    project_root: str | Path,
    config: Mapping[str, Any],
    *,
    inputs: Sequence[str | Path] = (),
    outputs: Sequence[str | Path] = (),
    source_metadata: Mapping[str, Any] | None = None,
    summaries: Mapping[str, Any] | None = None,
    status: str = "complete",
    overwrite: bool = True,
) -> Path:
    """Write a content-hashed record for one pipeline stage."""

    root = Path(project_root).resolve()
    target = configured_path(root, config, "provenance_stages") / f"{stage}.json"
    payload = {
        "stage": str(stage),
        "status": str(status),
        "generated_utc": utc_now(),
        "config_sha256": config_sha256(root),
        "inputs": artifact_records(inputs, root),
        "outputs": artifact_records(outputs, root),
        "source_metadata": dict(source_metadata or {}),
        "summaries": dict(summaries or {}),
    }
    return write_json(payload, target, overwrite=overwrite)


def write_build_provenance(
    name: str,
    payload: Mapping[str, Any],
    project_root: str | Path,
    config: Mapping[str, Any],
    *,
    overwrite: bool = True,
) -> Path:
    """Write a build-level record under the configured provenance directory."""

    root = Path(project_root).resolve()
    target = configured_path(root, config, "provenance_builds") / f"{name}.json"
    normalized = {
        "name": str(name),
        "generated_utc": utc_now(),
        "config_sha256": config_sha256(root),
        **dict(payload),
    }
    return write_json(normalized, target, overwrite=overwrite)


def stage_provenance_path(
    stage: str,
    project_root: str | Path,
    config: Mapping[str, Any],
) -> Path:
    """Return the canonical provenance record for one stage."""

    return (
        configured_path(project_root, config, "provenance_stages")
        / f"{stage}.json"
    )


def require_completed_stage(
    stage: str,
    project_root: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Require a complete predecessor and verify every recorded output hash."""

    target = stage_provenance_path(stage, project_root, config)
    if not target.is_file():
        raise RuntimeError(f"Required predecessor provenance is absent: {target}")
    with target.open("r", encoding="utf-8") as handle:
        record = json.load(handle)
    if record.get("status") != "complete":
        raise RuntimeError(
            f"Required predecessor {stage} is {record.get('status')!r}, "
            "so downstream execution is forbidden."
        )
    if record.get("config_sha256") != config_sha256(project_root):
        raise RuntimeError(
            f"Configuration changed after predecessor {stage} completed."
        )
    root = Path(project_root).resolve()
    for output in record.get("outputs", []):
        path = Path(str(output["path"]))
        target_path = path if path.is_absolute() else root / path
        if not target_path.is_file():
            raise RuntimeError(
                f"Predecessor {stage} output is absent: {target_path}"
            )
        observed = sha256_file(target_path)
        if observed != output.get("sha256"):
            raise RuntimeError(
                f"Predecessor {stage} output hash changed: {target_path}"
            )
    return record
