from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

from src.config_files.configs import ClassifierSpec


ClassifierFactory = Callable[[int], object]
FEDERATED_CLASSIFIER_NAMES = {"fedavg_logistic_regression"}


class FedAvgLogisticRegression:
    def __init__(
        self,
        random_state: int,
        num_rounds: int = 10,
        local_epochs: int = 1,
        learning_rate: float = 0.05,
        batch_size: int = 256,
        l2_penalty: float = 1e-4,
        average_by: str = "samples",
        shuffle: bool = True,
    ) -> None:
        self.random_state = random_state
        self.num_rounds = max(1, int(num_rounds))
        self.local_epochs = max(1, int(local_epochs))
        self.learning_rate = float(learning_rate)
        self.batch_size = max(1, int(batch_size))
        self.l2_penalty = float(l2_penalty)
        self.average_by = average_by
        self.shuffle = shuffle
        self.classes_ = np.asarray([0, 1], dtype=np.int64)
        self.coef_: np.ndarray | None = None
        self.intercept_: np.ndarray | None = None

    def _run_local_training(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None,
        coef: np.ndarray,
        intercept: np.ndarray,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        local_coef = coef.astype(np.float64, copy=True)
        local_intercept = intercept.astype(np.float64, copy=True)
        rng = np.random.default_rng(seed)
        num_rows = len(X)

        for _ in range(self.local_epochs):
            indices = np.arange(num_rows)
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, num_rows, self.batch_size):
                batch_idx = indices[start : start + self.batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx].astype(np.float64)
                weight_batch = None if sample_weight is None else sample_weight[batch_idx].astype(np.float64)

                logits = (X_batch @ local_coef) + local_intercept[0]
                probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
                errors = probabilities - y_batch

                if weight_batch is not None:
                    errors = errors * weight_batch
                    normalizer = max(float(weight_batch.sum()), 1.0)
                else:
                    normalizer = float(len(batch_idx))

                gradient_w = (X_batch.T @ errors) / normalizer
                gradient_w += self.l2_penalty * local_coef
                gradient_b = np.asarray([errors.sum() / normalizer], dtype=np.float64)
                local_coef -= self.learning_rate * gradient_w
                local_intercept -= self.learning_rate * gradient_b

        return local_coef, local_intercept

    def fit_federated(self, client_datasets: list[dict[str, np.ndarray]]) -> "FedAvgLogisticRegression":
        valid_datasets = [dataset for dataset in client_datasets if len(dataset["X"]) > 0]
        if not valid_datasets:
            raise ValueError("FedAvg logistic regression requires at least one non-empty client dataset")

        num_features = valid_datasets[0]["X"].shape[1]
        global_coef = np.zeros(num_features, dtype=np.float64)
        global_intercept = np.zeros(1, dtype=np.float64)

        for round_index in range(self.num_rounds):
            local_updates: list[tuple[float, np.ndarray, np.ndarray]] = []
            for client_index, dataset in enumerate(valid_datasets):
                X_client = dataset["X"].astype(np.float64, copy=False)
                y_client = dataset["y"].astype(np.int64, copy=False)
                sample_weight = dataset.get("sample_weight")
                if sample_weight is not None:
                    sample_weight = sample_weight.astype(np.float64, copy=False)

                local_coef, local_intercept = self._run_local_training(
                    X=X_client,
                    y=y_client,
                    sample_weight=sample_weight,
                    coef=global_coef,
                    intercept=global_intercept,
                    seed=self.random_state + (round_index * 1009) + client_index,
                )
                if self.average_by == "sample_weight" and sample_weight is not None:
                    aggregation_weight = max(float(sample_weight.sum()), 1.0)
                else:
                    aggregation_weight = float(len(X_client))
                local_updates.append((aggregation_weight, local_coef, local_intercept))

            total_weight = sum(weight for weight, _, _ in local_updates)
            global_coef = sum(weight * coef for weight, coef, _ in local_updates) / total_weight
            global_intercept = sum(weight * intercept for weight, _, intercept in local_updates) / total_weight

        self.coef_ = global_coef.reshape(1, -1)
        self.intercept_ = global_intercept
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise ValueError("FedAvg logistic regression must be trained before scoring")
        return (X @ self.coef_[0]) + self.intercept_[0]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self.decision_function(X)
        positive_probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        negative_probability = 1.0 - positive_probability
        return np.column_stack([negative_probability, positive_probability])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int64)


def get_model_registry(specs: list[ClassifierSpec]) -> dict[str, ClassifierFactory]:
    registry: dict[str, ClassifierFactory] = {}
    for spec in specs:
        params = dict(spec.params)
        if spec.name == "logistic_regression":
            registry[spec.name] = lambda seed, params=params: LogisticRegression(
                random_state=seed,
                **params,
            )
        elif spec.name == "fedavg_logistic_regression":
            registry[spec.name] = lambda seed, params=params: FedAvgLogisticRegression(
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
        elif spec.name == "lightgbm":
            def build_lightgbm(seed: int, params=params):
                if LGBMClassifier is None:
                    raise ImportError(
                        "LightGBM is not installed. Update the environment from environment.yml "
                        "or install the 'lightgbm' package before using the lightgbm classifier."
                    )
                return LGBMClassifier(
                    random_state=seed,
                    n_jobs=-1,
                    **params,
                )

            registry[spec.name] = build_lightgbm
    return registry


def is_federated_classifier(model_name: str) -> bool:
    return model_name in FEDERATED_CLASSIFIER_NAMES


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
