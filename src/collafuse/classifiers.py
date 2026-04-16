from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.config_files.configs import ClassifierSpec


ClassifierFactory = Callable[[int], object]


def get_model_registry(specs: list[ClassifierSpec]) -> dict[str, ClassifierFactory]:
    registry: dict[str, ClassifierFactory] = {}
    for spec in specs:
        params = dict(spec.params)
        if spec.name == "logistic_regression":
            registry[spec.name] = lambda seed, params=params: LogisticRegression(
                random_state=seed,
                **params,
            )
        elif spec.name == "random_forest":
            registry[spec.name] = lambda seed, params=params: RandomForestClassifier(
                random_state=seed,
                n_jobs=-1,
                **params,
            )
        elif spec.name == "hist_gradient_boosting":
            registry[spec.name] = lambda seed, params=params: HistGradientBoostingClassifier(
                random_state=seed,
                **params,
            )
    return registry


def build_sample_weights(y: np.ndarray) -> np.ndarray:
    positives = max(1, int(y.sum()))
    negatives = max(1, len(y) - positives)
    positive_weight = negatives / positives
    return np.where(y == 1, positive_weight, 1.0).astype(np.float32)


def predict_scores(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        shifted = raw - raw.min()
        scale = shifted.max()
        return shifted / scale if scale > 0 else np.zeros_like(shifted)
    return model.predict(X)
