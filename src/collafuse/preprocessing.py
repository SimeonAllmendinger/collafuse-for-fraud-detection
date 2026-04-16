from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from src.config_files.configs import ClientSplitConfig, DataConfig
from src.pipeline.common import ensure_directory


@dataclass
class PreprocessingTemplate:
    categorical_columns: list[str]
    numeric_columns: list[str]
    encoded_columns: list[str]
    drop_columns_corr: list[str]
    drop_columns_lowvar: list[str]
    drop_columns_nan: list[str]
    fill_values: dict[str, float]
    final_columns: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def merge_raw_ieee_tables(
    transaction_path: str | Path,
    identity_path: str | Path,
    transaction_id_column: str,
) -> pd.DataFrame:
    transaction_df = pd.read_csv(transaction_path)
    identity_df = pd.read_csv(identity_path)
    return transaction_df.merge(identity_df, how="left", on=transaction_id_column)


def _slugify_client_label(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "other"


def _assign_clients_by_card_rules(frame: pd.DataFrame, config: ClientSplitConfig) -> pd.DataFrame:
    mapped = frame.copy()
    fallback = config.fallback_rule
    rule_map = {
        (
            (rule.card4 or "").strip().lower(),
            (rule.card6 or "").strip().lower(),
        ): rule
        for rule in config.rules
        if not rule.fallback
    }

    card4_values = mapped[config.card4_column].fillna("").astype(str).str.strip().str.lower()
    card6_values = mapped[config.card6_column].fillna("").astype(str).str.strip().str.lower()

    client_ids: list[str] = []
    client_labels: list[str] = []
    for card4, card6 in zip(card4_values, card6_values):
        rule = rule_map.get((card4, card6), fallback)
        client_ids.append(rule.client_id)
        client_labels.append(rule.label)

    mapped[config.client_column] = client_ids
    mapped[config.client_label_column] = client_labels
    return mapped


def _assign_clients_by_categorical_column(frame: pd.DataFrame, config: ClientSplitConfig) -> pd.DataFrame:
    if config.source_column is None:
        raise ValueError("source_column is required for categorical client assignment")

    mapped = frame.copy()
    values = mapped[config.source_column].fillna("__missing__").astype(str).str.strip()
    selected_values = config.categorical_values or values.value_counts().index.tolist()[: config.num_clients]
    client_map = {
        value: (
            f"CLIENT_{index}",
            f"{config.source_column}_{_slugify_client_label(value)}",
        )
        for index, value in enumerate(selected_values)
    }

    fallback_client_id = f"CLIENT_{len(client_map)}"
    fallback_client_label = f"{config.source_column}_other"
    assigned = [client_map.get(value, (fallback_client_id, fallback_client_label)) for value in values]
    mapped[config.client_column] = [client_id for client_id, _ in assigned]
    mapped[config.client_label_column] = [client_label for _, client_label in assigned]
    return mapped


def _assign_clients_by_quantile(frame: pd.DataFrame, config: ClientSplitConfig) -> pd.DataFrame:
    if config.source_column is None:
        raise ValueError("source_column is required for quantile client assignment")

    mapped = frame.copy()
    source_values = pd.to_numeric(mapped[config.source_column], errors="coerce")
    if source_values.notna().sum() == 0:
        raise ValueError(f"client split source column '{config.source_column}' does not contain numeric values")

    filled_values = source_values.fillna(source_values.median())
    quantile_count = min(config.num_clients, max(1, len(filled_values)))
    rank_values = filled_values.rank(method="first")
    bins = pd.qcut(rank_values, q=quantile_count, labels=False, duplicates="drop")
    if not isinstance(bins, pd.Series):
        bins = pd.Series(bins, index=mapped.index)
    bins = bins.astype(int)

    mapped[config.client_column] = bins.map(lambda value: f"CLIENT_{value}")
    mapped[config.client_label_column] = bins.map(lambda value: f"{config.source_column}_quantile_{value}")
    return mapped


def assign_clients(frame: pd.DataFrame, config: ClientSplitConfig) -> pd.DataFrame:
    if config.strategy == "card_rules":
        return _assign_clients_by_card_rules(frame, config)
    if config.strategy == "categorical":
        return _assign_clients_by_categorical_column(frame, config)
    if config.strategy == "quantile":
        return _assign_clients_by_quantile(frame, config)
    raise ValueError(f"Unsupported client split strategy: {config.strategy}")


def _split_feature_columns(frame: pd.DataFrame, excluded_columns: set[str]) -> tuple[list[str], list[str]]:
    feature_frame = frame.drop(
        columns=[column for column in excluded_columns if column in frame.columns], errors="ignore"
    )
    categorical_columns = feature_frame.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
    numeric_columns = [column for column in feature_frame.columns if column not in categorical_columns]
    return categorical_columns, numeric_columns


def _encode_categorical_columns(
    frame: pd.DataFrame,
    categorical_columns: list[str],
    encoder: OneHotEncoder | None
) -> pd.DataFrame:
    if not categorical_columns or encoder is None:
        return pd.DataFrame(index=frame.index)

    categorical_frame = frame[categorical_columns].copy().fillna("__nan__").astype(str)
    encoded = encoder.transform(categorical_frame)
    return pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_columns), index=frame.index)


def _compute_sparse_row_mask(
    frame: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
    threshold: float
) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=bool)

    numeric_frame = frame.reindex(columns=numeric_columns).apply(pd.to_numeric, errors="coerce")
    categorical_frame = frame.reindex(columns=categorical_columns)
    raw_feature_frame = pd.concat([numeric_frame, categorical_frame], axis=1)
    row_nan_ratio = raw_feature_frame.isna().mean(axis=1)
    return row_nan_ratio <= threshold


def drop_highly_correlated_columns(frame: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, list[str]]:
    working = frame.copy()
    dropped: list[str] = []
    while True:
        corr = working.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        candidates = [
            (column_a, column_b)
            for column_a in upper.columns
            for column_b in upper.index
            if pd.notnull(upper.loc[column_b, column_a]) and upper.loc[column_b, column_a] > threshold
        ]
        if not candidates:
            break
        to_drop: set[str] = set()
        for column_a, column_b in candidates:
            drop_column = column_a if working[column_a].var() < working[column_b].var() else column_b
            to_drop.add(drop_column)
        working = working.drop(columns=list(to_drop))
        dropped.extend(sorted(to_drop))
    return working, sorted(set(dropped))


def get_low_variance_columns(frame: pd.DataFrame, threshold: float) -> list[str]:
    if frame.empty:
        return []
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(frame)
    return frame.columns[~selector.get_support()].tolist()


def get_high_nan_columns(frame: pd.DataFrame, threshold: float) -> list[str]:
    nan_ratio = frame.isna().mean()
    return nan_ratio[nan_ratio > threshold].index.tolist()


def get_fill_values(frame: pd.DataFrame, strategy: str) -> dict[str, float]:
    fill_values: dict[str, float] = {}
    for column in frame.columns:
        if strategy == "mean":
            fill_values[column] = float(frame[column].mean()) if frame[column].isna().any() else 0.0
        elif strategy == "median":
            fill_values[column] = float(frame[column].median()) if frame[column].isna().any() else 0.0
        else:
            fill_values[column] = 0.0
    return fill_values


def fit_global_template(
    reference_frame: pd.DataFrame,
    data_config: DataConfig,
    client_config: ClientSplitConfig,
) -> tuple[PreprocessingTemplate, OneHotEncoder | None, MinMaxScaler]:
    excluded = {
        data_config.transaction_id_column,
        data_config.label_column,
        client_config.client_column,
        client_config.client_label_column,
        *data_config.drop_columns,
    }
    categorical_columns, numeric_columns = _split_feature_columns(reference_frame, excluded)
    sparse_mask = _compute_sparse_row_mask(
        reference_frame,
        numeric_columns,
        categorical_columns,
        data_config.row_nan_threshold
    )
    filtered_reference = reference_frame.loc[sparse_mask].reset_index(drop=True)
    if filtered_reference.empty:
        raise ValueError(
            "Reference client preprocessing removed every row after sparse-row filtering. "
            "Increase data.row_nan_threshold or review missingness in the raw data."
        )

    encoder: OneHotEncoder | None = None
    encoded_columns: list[str] = []
    if categorical_columns:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoder.fit(filtered_reference[categorical_columns].fillna("__nan__").astype(str))
        encoded_columns = encoder.get_feature_names_out(categorical_columns).tolist()

    numeric_frame = filtered_reference.reindex(columns=numeric_columns).apply(pd.to_numeric, errors="coerce")
    encoded_frame = _encode_categorical_columns(filtered_reference, categorical_columns, encoder)
    combined = pd.concat([numeric_frame.reset_index(drop=True), encoded_frame.reset_index(drop=True)], axis=1)

    corr_dropped, drop_corr = drop_highly_correlated_columns(combined, data_config.correlation_threshold)
    low_variance_columns = get_low_variance_columns(corr_dropped, data_config.variance_threshold)
    variance_dropped = corr_dropped.drop(columns=low_variance_columns, errors="ignore")
    high_nan_columns = get_high_nan_columns(variance_dropped, data_config.col_nan_threshold)
    nan_dropped = variance_dropped.drop(columns=high_nan_columns, errors="ignore")
    fill_values = get_fill_values(nan_dropped, data_config.fill_strategy)
    filled = nan_dropped.fillna(fill_values)
    deduplicated = filled.drop_duplicates().reset_index(drop=True)
    scaler = MinMaxScaler()
    scaler.fit(deduplicated if not deduplicated.empty else filled)

    template = PreprocessingTemplate(
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        encoded_columns=encoded_columns,
        drop_columns_corr=drop_corr,
        drop_columns_lowvar=low_variance_columns,
        drop_columns_nan=high_nan_columns,
        fill_values=fill_values,
        final_columns=filled.columns.tolist(),
    )
    return template, encoder, scaler


def apply_global_template(
    frame: pd.DataFrame,
    template: PreprocessingTemplate,
    encoder: OneHotEncoder | None,
    scaler: MinMaxScaler,
    data_config: DataConfig,
    client_config: ClientSplitConfig,
) -> pd.DataFrame:
    excluded = {
        data_config.transaction_id_column,
        data_config.label_column,
        client_config.client_column,
        client_config.client_label_column,
        *data_config.drop_columns,
    }
    feature_frame = frame.drop(columns=[column for column in excluded if column in frame.columns], errors="ignore")
    sparse_mask = _compute_sparse_row_mask(
        feature_frame,
        template.numeric_columns,
        template.categorical_columns,
        data_config.row_nan_threshold
    )
    filtered_frame = frame.loc[sparse_mask].reset_index(drop=True)
    filtered_feature_frame = feature_frame.loc[sparse_mask].reset_index(drop=True)

    numeric_frame = filtered_feature_frame.reindex(
        columns=template.numeric_columns
    ).apply(pd.to_numeric, errors="coerce")
    encoded_frame = _encode_categorical_columns(filtered_feature_frame, template.categorical_columns, encoder)
    encoded_frame = encoded_frame.reindex(columns=template.encoded_columns, fill_value=0.0)
    combined = pd.concat([numeric_frame.reset_index(drop=True), encoded_frame.reset_index(drop=True)], axis=1)

    drop_columns = set(template.drop_columns_corr + template.drop_columns_lowvar + template.drop_columns_nan)
    combined = combined.drop(columns=[column for column in drop_columns if column in combined.columns], errors="ignore")
    combined = combined.reindex(columns=template.final_columns).copy()
    filled = combined.fillna(template.fill_values).fillna(0.0).copy()
    labels = filtered_frame[data_config.label_column].reset_index(drop=True).astype(int)
    label_frame = labels.rename(data_config.label_column).to_frame()
    filled_with_labels = pd.concat([filled, label_frame], axis=1)
    deduplicated = filled_with_labels.drop_duplicates(subset=template.final_columns).reset_index(drop=True)

    scaled_features = pd.DataFrame(
        scaler.transform(deduplicated[template.final_columns]),
        columns=template.final_columns,
    ).copy()
    scaled_labels = deduplicated[data_config.label_column].astype(int).rename(data_config.label_column).to_frame()
    return pd.concat([scaled_features, scaled_labels], axis=1)


def split_client_frame(frame: pd.DataFrame, data_config: DataConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify = frame[data_config.label_column] if frame[data_config.label_column].nunique() > 1 else None
    try:
        train_frame, test_frame = train_test_split(
            frame,
            test_size=data_config.test_size,
            random_state=data_config.shuffle_seed,
            stratify=stratify,
        )
    except ValueError:
        train_frame, test_frame = train_test_split(
            frame,
            test_size=data_config.test_size,
            random_state=data_config.shuffle_seed,
            stratify=None,
        )
    return train_frame.reset_index(drop=True), test_frame.reset_index(drop=True)


def save_template_artifacts(
    output_dir: str | Path,
    template: PreprocessingTemplate,
    encoder: OneHotEncoder | None,
    scaler: MinMaxScaler,
) -> dict[str, str]:
    output_dir = ensure_directory(output_dir)
    template_path = output_dir / "column_template.json"
    with template_path.open("w", encoding="utf-8") as handle:
        json.dump(template.to_dict(), handle, indent=2)

    paths = {"template_path": str(template_path)}
    if encoder is not None:
        encoder_path = output_dir / "global_encoder.pkl"
        joblib.dump(encoder, encoder_path)
        paths["encoder_path"] = str(encoder_path)
    scaler_path = output_dir / "global_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    paths["scaler_path"] = str(scaler_path)
    return paths
