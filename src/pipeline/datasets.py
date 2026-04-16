from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.collafuse.preprocessing import merge_raw_ieee_tables
from src.config_files.configs import DataConfig, PaperPipelineConfig

LOGGER = logging.getLogger("collafuse")


@dataclass
class LoadedDataset:
    frame: pd.DataFrame
    source_paths: dict[str, str]


def _read_table(path: str | Path, *, header: int | None = 0) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, header=header)


def _coerce_binary_label(frame: pd.DataFrame, label_column: str) -> pd.DataFrame:
    normalized = frame.copy()
    normalized[label_column] = pd.to_numeric(normalized[label_column], errors="coerce")
    normalized = normalized.loc[normalized[label_column].notna()].reset_index(drop=True)
    normalized[label_column] = normalized[label_column].astype(int)
    return normalized


def _ensure_transaction_id(
    frame: pd.DataFrame,
    data_config: DataConfig,
    rename_from: str | None = None,
) -> pd.DataFrame:
    normalized = frame.copy()
    if rename_from and rename_from in normalized.columns and rename_from != data_config.transaction_id_column:
        normalized = normalized.rename(columns={rename_from: data_config.transaction_id_column})
    if data_config.transaction_id_column not in normalized.columns:
        normalized[data_config.transaction_id_column] = np.arange(1, len(normalized) + 1)
    return normalized


def _resolve_required_path(path_value: Path | None, label: str) -> Path:
    if path_value is None:
        raise ValueError(f"{label} is required for the selected dataset")
    return Path(path_value)


def _load_ieee_cis(
    config: PaperPipelineConfig,
    raw_transaction_path: str | Path | None,
    raw_identity_path: str | Path | None,
) -> LoadedDataset:
    transaction_path = Path(raw_transaction_path) if raw_transaction_path else config.paths.raw_transaction_path
    identity_path = Path(raw_identity_path) if raw_identity_path else config.paths.raw_identity_path
    frame = merge_raw_ieee_tables(transaction_path, identity_path, config.data.transaction_id_column)
    frame = _coerce_binary_label(frame, config.data.label_column)
    return LoadedDataset(
        frame=frame,
        source_paths={
            "raw_transaction_path": str(transaction_path),
            "raw_identity_path": str(identity_path),
        },
    )


def _load_paysim(config: PaperPipelineConfig, raw_main_path: str | Path | None) -> LoadedDataset:
    main_path = Path(raw_main_path) if raw_main_path else _resolve_required_path(config.paths.raw_main_path, "paths.raw_main_path")
    frame = _read_table(main_path)
    if "isFraud" in frame.columns and config.data.label_column != "isFraud":
        frame = frame.rename(columns={"isFraud": config.data.label_column})
    frame = _ensure_transaction_id(frame, config.data)
    frame = _coerce_binary_label(frame, config.data.label_column)
    return LoadedDataset(frame=frame, source_paths={"raw_main_path": str(main_path)})


def _load_credit_card_fraud(config: PaperPipelineConfig, raw_main_path: str | Path | None) -> LoadedDataset:
    main_path = Path(raw_main_path) if raw_main_path else _resolve_required_path(config.paths.raw_main_path, "paths.raw_main_path")
    frame = _read_table(main_path)
    if "Class" in frame.columns and config.data.label_column != "Class":
        frame = frame.rename(columns={"Class": config.data.label_column})
    frame = _ensure_transaction_id(frame, config.data)
    frame = _coerce_binary_label(frame, config.data.label_column)
    return LoadedDataset(frame=frame, source_paths={"raw_main_path": str(main_path)})


def _load_baf(config: PaperPipelineConfig, raw_main_path: str | Path | None) -> LoadedDataset:
    main_path = Path(raw_main_path) if raw_main_path else _resolve_required_path(config.paths.raw_main_path, "paths.raw_main_path")
    frame = _read_table(main_path)
    if "fraud_bool" in frame.columns and config.data.label_column != "fraud_bool":
        frame = frame.rename(columns={"fraud_bool": config.data.label_column})
    frame = _ensure_transaction_id(frame, config.data)
    frame = _coerce_binary_label(frame, config.data.label_column)
    return LoadedDataset(frame=frame, source_paths={"raw_main_path": str(main_path)})


def _load_elliptic(
    config: PaperPipelineConfig,
    raw_main_path: str | Path | None,
    raw_aux_path: str | Path | None,
    raw_edge_path: str | Path | None,
) -> LoadedDataset:
    features_path = Path(raw_main_path) if raw_main_path else _resolve_required_path(config.paths.raw_main_path, "paths.raw_main_path")
    classes_path = Path(raw_aux_path) if raw_aux_path else _resolve_required_path(config.paths.raw_aux_path, "paths.raw_aux_path")
    edge_path = Path(raw_edge_path) if raw_edge_path else config.paths.raw_edge_path

    features = _read_table(features_path, header=None)
    if features.shape[1] < 3:
        raise ValueError("Elliptic features table must contain txId, time_step, and at least one feature column")

    features.columns = ["txId", "time_step", *[f"feature_{index}" for index in range(features.shape[1] - 2)]]
    classes = _read_table(classes_path)
    if "txId" not in classes.columns or "class" not in classes.columns:
        raise ValueError("Elliptic classes table must contain txId and class columns")

    frame = features.merge(classes, on="txId", how="inner")
    label_values = frame["class"].astype(str).str.strip().str.lower()
    frame = frame.loc[label_values.isin({"1", "2"})].reset_index(drop=True)
    normalized_labels = frame["class"].astype(str).str.strip().str.lower().map({"1": 1, "2": 0}).astype(int)
    if config.data.label_column == "class":
        frame["class"] = normalized_labels
    else:
        frame[config.data.label_column] = normalized_labels
        frame = frame.drop(columns=["class"])
    frame = _ensure_transaction_id(frame, config.data, rename_from="txId")
    if edge_path is not None:
        LOGGER.info("Elliptic edge list provided at %s but is not used by the current tabular pipeline", edge_path)
    return LoadedDataset(
        frame=frame,
        source_paths={
            "raw_main_path": str(features_path),
            "raw_aux_path": str(classes_path),
            **({"raw_edge_path": str(edge_path)} if edge_path is not None else {}),
        },
    )


def load_preparation_dataset(
    config: PaperPipelineConfig,
    raw_transaction_path: str | Path | None = None,
    raw_identity_path: str | Path | None = None,
    raw_main_path: str | Path | None = None,
    raw_aux_path: str | Path | None = None,
    raw_edge_path: str | Path | None = None,
) -> LoadedDataset:
    dataset_name = config.data.dataset_name
    LOGGER.info("Loading raw dataset for %s", dataset_name)
    if dataset_name == "ieee_cis":
        return _load_ieee_cis(config, raw_transaction_path, raw_identity_path)
    if dataset_name == "paysim":
        return _load_paysim(config, raw_main_path)
    if dataset_name == "credit_card_fraud":
        return _load_credit_card_fraud(config, raw_main_path)
    if dataset_name == "baf":
        return _load_baf(config, raw_main_path)
    if dataset_name == "elliptic":
        return _load_elliptic(config, raw_main_path, raw_aux_path, raw_edge_path)
    raise ValueError(f"Unsupported dataset: {dataset_name}")
