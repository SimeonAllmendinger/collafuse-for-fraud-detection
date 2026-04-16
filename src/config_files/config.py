from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from src.config_files.configs import PaperPipelineConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger("collafuse")


def resolve_project_path(path_value: Path) -> Path:
    return path_value if path_value.is_absolute() else (PROJECT_ROOT / path_value).resolve()


def resolve_optional_project_path(path_value: Path | None) -> Path | None:
    if path_value is None:
        return None
    return resolve_project_path(path_value)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(base_value, value)
        else:
            merged[key] = value
    return merged


def _resolve_config_ref(config_path: Path, ref: str) -> Path:
    ref_path = Path(ref)
    if ref_path.is_absolute():
        return ref_path
    return (config_path.parent / ref_path).resolve()


def _load_raw_config(config_path: Path, visited: set[Path] | None = None) -> dict[str, Any]:
    resolved_path = config_path.resolve()
    visited = visited or set()
    if resolved_path in visited:
        raise ValueError(f"Config extends cycle detected at {resolved_path}")
    visited.add(resolved_path)

    with resolved_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}
    if not isinstance(raw_config, dict):
        raise ValueError(f"Config file must contain a mapping at top level: {resolved_path}")

    extends_value = raw_config.pop("extends", None)
    if extends_value is None:
        return raw_config

    refs = [extends_value] if isinstance(extends_value, str) else list(extends_value)
    merged_base: dict[str, Any] = {}
    for ref in refs:
        base_path = _resolve_config_ref(resolved_path, ref)
        merged_base = _deep_merge(merged_base, _load_raw_config(base_path, visited.copy()))
    return _deep_merge(merged_base, raw_config)


def load_pipeline_config(config_path: str | Path) -> PaperPipelineConfig:
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()

    raw_config = _load_raw_config(config_path)
    config = PaperPipelineConfig.model_validate(raw_config)
    config.paths.raw_transaction_path = resolve_project_path(config.paths.raw_transaction_path)
    config.paths.raw_identity_path = resolve_project_path(config.paths.raw_identity_path)
    config.paths.raw_main_path = resolve_optional_project_path(config.paths.raw_main_path)
    config.paths.raw_aux_path = resolve_optional_project_path(config.paths.raw_aux_path)
    config.paths.raw_edge_path = resolve_optional_project_path(config.paths.raw_edge_path)
    config.paths.prepared_root = resolve_project_path(config.paths.prepared_root)
    config.paths.stage1_root = resolve_project_path(config.paths.stage1_root)
    config.paths.stage2_root = resolve_project_path(config.paths.stage2_root)
    config.paths.run_all_stages_root = resolve_project_path(config.paths.run_all_stages_root)
    return config
