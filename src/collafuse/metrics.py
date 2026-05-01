from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from src.collafuse.evaluation import evaluate_mmd


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.5,
        "average_precision": average_precision_score(
            y_true, y_score
        ) if len(np.unique(y_true)) > 1 else float(y_true.mean()),
    }


def aggregate_metrics(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    numeric_columns = ["precision", "recall", "f1", "roc_auc", "average_precision"]
    grouped = frame.groupby(group_columns, dropna=False)[numeric_columns].agg(["mean", "std"]).reset_index()
    grouped.columns = [
        "_".join(column).strip("_") if isinstance(column, tuple) else column
        for column in grouped.columns.to_flat_index()
    ]
    return grouped


def build_pca_tsne_embedding(
    frame: pd.DataFrame,
    feature_columns: list[str],
    pca_components: int,
    perplexity: float,
    learning_rate: float | str,
    random_state: int,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    feature_matrix = frame[feature_columns].to_numpy(dtype=np.float32)
    n_samples = len(feature_matrix)
    n_components = min(pca_components, feature_matrix.shape[1], max(1, n_samples - 1))
    reduced = PCA(n_components=n_components, random_state=random_state).fit_transform(feature_matrix)
    adjusted_perplexity = min(perplexity, max(1.0, float(n_samples - 1) / 3.0))
    embedded = TSNE(
        n_components=2,
        perplexity=adjusted_perplexity,
        learning_rate=learning_rate,
        init="pca",
        random_state=random_state,
    ).fit_transform(reduced)
    result = frame.copy()
    result["tsne_x"] = embedded[:, 0]
    result["tsne_y"] = embedded[:, 1]
    return result


def compute_mmd_rows(
    real_samples: np.ndarray,
    generated_samples: np.ndarray,
    seeds: list[int],
    batch_size: int,
    source: str,
    client_id: str,
    comparison_scope: str = "client",
) -> list[dict[str, Any]]:
    if len(real_samples) == 0 or len(generated_samples) == 0:
        return []

    sample_limit = min(batch_size, len(real_samples), len(generated_samples))
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        real_idx = rng.choice(len(real_samples), size=sample_limit, replace=len(real_samples) < sample_limit)
        synth_idx = rng.choice(
            len(generated_samples),
            size=sample_limit,
            replace=len(generated_samples) < sample_limit,
        )
        score = evaluate_mmd(
            real_samples=real_samples[real_idx],
            generated_samples=generated_samples[synth_idx],
            batch_size=sample_limit,
            seed=seed,
        )
        rows.append(
            {
                "comparison_scope": comparison_scope,
                "client_id": client_id,
                "synthetic_source": source,
                "seed": seed,
                "mmd": score,
            }
        )
    return rows
