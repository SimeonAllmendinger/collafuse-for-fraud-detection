from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def make_run_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path)
    ensure_directory(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return output_path


def serialize_config(config: Any) -> dict[str, Any]:
    if hasattr(config, "model_dump"):
        return config.model_dump(mode="json")
    raise TypeError("Config object does not support model_dump(mode='json')")


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def set_random_seed(seed: int) -> int:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def select_torch_device(preference: str) -> torch.device:
    import torch

    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preference == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_stage1_run(stage1_root: Path, stage1_run: str | None = None) -> Path:
    stage1_root = Path(stage1_root)

    if stage1_run:
        candidate = Path(stage1_run)
        resolved = candidate.resolve() if candidate.exists() else (stage1_root / stage1_run).resolve()
        if not resolved.exists():
            raise ValueError(f"Stage 1 run does not exist: {stage1_run}")
        if not (resolved / "run_manifest.json").exists():
            raise ValueError(f"Stage 1 run is missing run_manifest.json: {resolved}")
        return resolved

    if not stage1_root.exists():
        raise ValueError(f"Stage 1 root does not exist: {stage1_root}")

    candidates = sorted(
        path for path in stage1_root.iterdir() if path.is_dir() and (path / "run_manifest.json").exists()
    )
    if not candidates:
        raise ValueError(f"No Stage 1 runs were found under: {stage1_root}")
    return candidates[-1].resolve()


def resolve_stage2_run(stage2_root: Path, stage2_run: str | None = None) -> Path:
    stage2_root = Path(stage2_root)

    if stage2_run:
        candidate = Path(stage2_run)
        resolved = candidate.resolve() if candidate.exists() else (stage2_root / stage2_run).resolve()
        if not resolved.exists():
            raise ValueError(f"Stage 2 run does not exist: {stage2_run}")
        if not (resolved / "run_manifest.json").exists():
            raise ValueError(f"Stage 2 run is missing run_manifest.json: {resolved}")
        return resolved

    if not stage2_root.exists():
        raise ValueError(f"Stage 2 root does not exist: {stage2_root}")

    candidates = sorted(
        path for path in stage2_root.iterdir() if path.is_dir() and (path / "run_manifest.json").exists()
    )
    if not candidates:
        raise ValueError(f"No Stage 2 runs were found under: {stage2_root}")
    return candidates[-1].resolve()
