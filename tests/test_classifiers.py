from __future__ import annotations

import unittest

import pandas as pd

from src.collafuse.classifiers import get_model_registry
from src.collafuse.metrics import aggregate_metrics
from src.config_files.configs import ClassifierSpec


class ClassifierTest(unittest.TestCase):
    def test_classifier_registry_and_metric_aggregation(self) -> None:
        registry = get_model_registry(
            [
                ClassifierSpec(name="logistic_regression"),
                ClassifierSpec(name="random_forest"),
                ClassifierSpec(name="hist_gradient_boosting"),
            ]
        )
        self.assertEqual(set(registry), {"logistic_regression", "random_forest", "hist_gradient_boosting"})

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


if __name__ == "__main__":
    unittest.main()
