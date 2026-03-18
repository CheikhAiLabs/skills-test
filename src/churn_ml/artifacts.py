from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def save_bundle(bundle_path: Path, payload: dict[str, Any]) -> None:
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, bundle_path)


def load_bundle(bundle_path: Path) -> dict[str, Any]:
    return joblib.load(bundle_path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_markdown_report(path: Path, report_lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(report_lines))
        handle.write("\n")


def write_sample_payload(path: Path, rows: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_json(path, orient="records", indent=2)


def find_latest_bundle(artifact_root: Path) -> Path:
    candidates = sorted(
        artifact_root.glob("*/model_bundle.joblib"),
        key=lambda candidate: candidate.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No model bundle found under {artifact_root}")
    return candidates[0]
