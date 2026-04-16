from __future__ import annotations

import logging

import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE
from tqdm.auto import tqdm

LOGGER = logging.getLogger("collafuse")


def compute_target_synthetic_count(labels: pd.Series, ratio: float) -> int:
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    gap = max(0, negatives - positives)
    return int(round(ratio * gap))


def sample_from_pool(pool: pd.DataFrame, n_samples: int, random_state: int) -> pd.DataFrame:
    if n_samples <= 0 or pool.empty:
        return pool.iloc[0:0].copy()
    replace = len(pool) < n_samples
    return pool.sample(n=n_samples, replace=replace, random_state=random_state).reset_index(drop=True)


def generate_random_oversampling_samples(
    train_df: pd.DataFrame,
    label_column: str,
    target_count: int,
    random_state: int,
    progress_desc: str | None = None,
) -> pd.DataFrame:
    if target_count <= 0:
        if progress_desc:
            LOGGER.info("%s: skipping because target_count=0", progress_desc)
        return train_df.iloc[0:0].copy()

    fraud_pool = train_df.loc[train_df[label_column] == 1].reset_index(drop=True)
    if progress_desc:
        LOGGER.info("%s: resampling %s fraud rows with replacement", progress_desc, target_count)
    sampled = sample_from_pool(fraud_pool, target_count, random_state)
    if progress_desc:
        LOGGER.info("%s: created %s synthetic samples", progress_desc, len(sampled))
    return sampled


def _materialize_oversampler_samples(
    train_df: pd.DataFrame,
    label_column: str,
    target_count: int,
    oversampler_name: str,
    random_state: int,
    n_neighbors: int,
    progress_desc: str | None = None,
) -> pd.DataFrame:
    if target_count <= 0:
        if progress_desc:
            LOGGER.info("%s: skipping because target_count=0", progress_desc)
        return train_df.iloc[0:0].copy()

    feature_columns = [column for column in train_df.columns if column != label_column]
    positives = int((train_df[label_column] == 1).sum())
    if positives < 2:
        if progress_desc:
            LOGGER.info("%s: skipping because only %s positive sample(s) are available", progress_desc, positives)
        return train_df.iloc[0:0].copy()

    effective_neighbors = max(1, min(n_neighbors, positives - 1))
    X_train = train_df[feature_columns]
    y_train = train_df[label_column]

    if oversampler_name == "smote":
        oversampler = SMOTE(sampling_strategy=1.0, random_state=random_state, k_neighbors=effective_neighbors)
    else:
        oversampler = ADASYN(sampling_strategy=1.0, random_state=random_state, n_neighbors=effective_neighbors)

    if progress_desc:
        LOGGER.info("%s: building %s synthetic samples", progress_desc, target_count)
    with tqdm(
        total=3,
        desc=progress_desc or f"{oversampler_name.upper()} samples",
        leave=True,
        disable=progress_desc is None
    ) as progress:
        try:
            X_resampled, y_resampled = oversampler.fit_resample(X_train, y_train)
        except (ValueError, RuntimeError):
            if progress_desc:
                LOGGER.warning("%s: oversampler failed, returning empty sample set", progress_desc)
            return train_df.iloc[0:0].copy()
        progress.update(1)

        n_new = len(X_resampled) - len(X_train)
        if n_new <= 0:
            return train_df.iloc[0:0].copy()

        synthetic = pd.DataFrame(X_resampled[-n_new:], columns=feature_columns)
        progress.update(1)
        label_frame = pd.DataFrame({label_column: y_resampled[-n_new:]}, index=synthetic.index)
        synthetic = pd.concat([synthetic.copy(), label_frame], axis=1)
        sampled = sample_from_pool(synthetic, target_count, random_state)
        progress.update(1)
    if progress_desc:
        LOGGER.info("%s: created %s synthetic samples", progress_desc, len(sampled))
    return sampled


def generate_smote_samples(
    train_df: pd.DataFrame,
    label_column: str,
    target_count: int,
    random_state: int,
    n_neighbors: int,
    progress_desc: str | None = None,
) -> pd.DataFrame:
    return _materialize_oversampler_samples(
        train_df,
        label_column,
        target_count,
        "smote",
        random_state,
        n_neighbors,
        progress_desc=progress_desc,
    )


def generate_adasyn_samples(
    train_df: pd.DataFrame,
    label_column: str,
    target_count: int,
    random_state: int,
    n_neighbors: int,
    progress_desc: str | None = None,
) -> pd.DataFrame:
    return _materialize_oversampler_samples(
        train_df,
        label_column,
        target_count,
        "adasyn",
        random_state,
        n_neighbors,
        progress_desc=progress_desc,
    )
