from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.collafuse.classifiers import get_model_registry
from src.collafuse.metrics import aggregate_metrics
from src.config_files.configs import ClassifierSpec


class ClassifierTest(unittest.TestCase):
    def test_classifier_registry_and_metric_aggregation(self) -> None:
        registry = get_model_registry(
            [
                ClassifierSpec(name="logistic_regression"),
                ClassifierSpec(name="fedavg_logistic_regression"),
                ClassifierSpec(name="random_forest"),
                ClassifierSpec(name="hist_gradient_boosting"),
            ]
        )
        self.assertEqual(
            set(registry),
            {"logistic_regression", "fedavg_logistic_regression", "random_forest", "hist_gradient_boosting"},
        )

        raw = pd.DataFrame(
            [
                {"model": "logistic_regression", "synthetic_source": "collafuse", "ratio": 0.2, "precision": 0.8, "recall": 0.7, "f1": 0.75, "roc_auc": 0.82, "average_precision": 0.81},
                {"model": "logistic_regression", "synthetic_source": "collafuse", "ratio": 0.2, "precision": 0.6, "recall": 0.9, "f1": 0.72, "roc_auc": 0.84, "average_precision": 0.83},
            ]
        )
        summary = aggregate_metrics(raw, ["model", "synthetic_source", "ratio"])
        self.assertIn("f1_mean", summary.columns)
        self.assertIn("f1_std", summary.columns)
        self.assertGreater(summary.loc[0, "f1_mean"], 0.7)

    def test_fedavg_logistic_regression_produces_probabilities(self) -> None:
        registry = get_model_registry(
            [
                ClassifierSpec(
                    name="fedavg_logistic_regression",
                    params={"num_rounds": 3, "local_epochs": 2, "learning_rate": 0.1, "batch_size": 2},
                )
            ]
        )
        model = registry["fedavg_logistic_regression"](7)
        model.fit_federated(
            [
                {
                    "X": np.asarray([[0.0, 0.0], [0.2, 0.1], [0.9, 1.1]], dtype=np.float32),
                    "y": np.asarray([0, 0, 1], dtype=np.int64),
                    "sample_weight": None,
                },
                {
                    "X": np.asarray([[1.0, 0.8], [1.2, 1.0], [0.1, 0.2]], dtype=np.float32),
                    "y": np.asarray([1, 1, 0], dtype=np.int64),
                    "sample_weight": np.asarray([1.0, 2.0, 1.0], dtype=np.float32),
                },
            ]
        )

        probabilities = model.predict_proba(np.asarray([[0.05, 0.1], [1.1, 0.9]], dtype=np.float32))[:, 1]

        self.assertEqual(probabilities.shape, (2,))
        self.assertTrue(np.all(probabilities >= 0.0))
        self.assertTrue(np.all(probabilities <= 1.0))
        self.assertLess(probabilities[0], probabilities[1])


if __name__ == "__main__":
    unittest.main()
