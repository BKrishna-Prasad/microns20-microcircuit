"""Atomic artifact writers, immutable raw snapshots, and content hashes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd


def require_columns(
    dataframe: pd.DataFrame,
    required: Iterable[str],
    name: str,
) -> None:
    """Raise a schema error listing missing and available columns."""

    missing = set(required) - set(dataframe.columns)
    if missing:
        raise ValueError(
            f"{name} is missing columns {sorted(missing)}; "
            f"available columns are {dataframe.columns.tolist()}."
        )


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _native(value: Any) -> Any:
    if value is pd.NA or value is None:
        return None
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_native(item) for item in value.tolist()]
    if isinstance(value, (list, tuple, set)):
        return [_native(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def dataframe_digest(dataframe: pd.DataFrame) -> str:
    """Return an order-independent logical SHA-256 for a dataframe."""

    rows = []
    columns = [str(column) for column in dataframe.columns]
    for record in dataframe.to_dict(orient="records"):
        normalized = {str(key): _native(value) for key, value in record.items()}
        rows.append(
            json.dumps(
                normalized,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
        )
    payload = json.dumps(
        {"columns": columns, "rows": sorted(rows)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_dataframe(path: str | Path) -> pd.DataFrame:
    """Read a supported tabular artifact."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported dataframe artifact: {path}")


def _temporary_path(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(descriptor)
    return Path(name)


def _write_dataframe_file(dataframe: pd.DataFrame, path: Path) -> None:
    if path.suffix == ".parquet":
        dataframe.to_parquet(path, index=False)
    elif path.suffix == ".csv":
        dataframe.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported dataframe artifact: {path}")


def write_dataframe(
    dataframe: pd.DataFrame,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write a derived table with explicit replacement semantics."""

    target = Path(path)
    if target.exists():
        existing = read_dataframe(target)
        if dataframe_digest(existing) == dataframe_digest(dataframe):
            return target
        if not overwrite:
            raise FileExistsError(
                f"Refusing to replace a different derived artifact: {target}"
            )

    temporary = _temporary_path(target)
    try:
        _write_dataframe_file(dataframe, temporary.with_suffix(target.suffix))
        written = temporary.with_suffix(target.suffix)
        os.replace(written, target)
    finally:
        if temporary.exists():
            temporary.unlink()
        sibling = temporary.with_suffix(target.suffix)
        if sibling.exists():
            sibling.unlink()
    return target


def write_raw_snapshot(dataframe: pd.DataFrame, path: str | Path) -> Path:
    """Write raw data once, or validate an existing logical snapshot."""

    target = Path(path)
    if target.exists():
        existing = read_dataframe(target)
        existing_digest = dataframe_digest(existing)
        incoming_digest = dataframe_digest(dataframe)
        if existing_digest != incoming_digest:
            raise RuntimeError(
                "Pinned raw artifact differs from the existing immutable "
                f"snapshot: {target}; existing={existing_digest}, "
                f"incoming={incoming_digest}."
            )
        return target
    return write_dataframe(dataframe, target, overwrite=False)


def write_text_once(text: str, path: str | Path) -> Path:
    """Create immutable raw text or verify that existing bytes agree."""

    target = Path(path)
    encoded = text.encode("utf-8")
    if target.exists():
        if target.read_bytes() != encoded:
            raise RuntimeError(
                f"Refusing to replace a different immutable raw file: {target}"
            )
        return target

    temporary = _temporary_path(target)
    try:
        temporary.write_bytes(encoded)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def write_json(
    payload: Mapping[str, Any],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write normalized JSON with explicit replacement semantics."""

    target = Path(path)
    encoded = (
        json.dumps(
            _native(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")
    if target.exists():
        if target.read_bytes() == encoded:
            return target
        if not overwrite:
            raise FileExistsError(f"Refusing to replace JSON artifact: {target}")

    temporary = _temporary_path(target)
    try:
        temporary.write_bytes(encoded)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def artifact_record(
    path: str | Path,
    project_root: str | Path,
    *,
    dataframe: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Describe an artifact for stage provenance."""

    root = Path(project_root).resolve()
    target = Path(path).resolve()
    try:
        display_path = str(target.relative_to(root))
    except ValueError:
        display_path = str(target)
    record: dict[str, Any] = {
        "path": display_path,
        "sha256": sha256_file(target),
        "size_bytes": target.stat().st_size,
    }
    if dataframe is not None:
        record.update(
            {
                "rows": int(len(dataframe)),
                "columns": [str(column) for column in dataframe.columns],
                "logical_sha256": dataframe_digest(dataframe),
            }
        )
    return record


def assert_directory_writable(path: str | Path) -> None:
    """Test directory writability without leaving a file behind."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".write_test.", dir=directory)
    os.close(descriptor)
    Path(name).unlink()
