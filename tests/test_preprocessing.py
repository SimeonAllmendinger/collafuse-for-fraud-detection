from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path

import pandas as pd

from src.pipeline.common import read_json
from src.pipeline.prepare import run_preparation
from tests.helpers import build_test_config


class PreprocessingTest(unittest.TestCase):
    def test_preparation_aligns_client_feature_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config = build_test_config(tmp_path)
            with warnings.catch_warnings():
                warnings.simplefilter("error", pd.errors.PerformanceWarning)
                prepared_root = run_preparation(config)
            metadata = read_json(prepared_root / "metadata.json")
            label_column = metadata["label_column"]
            feature_columns = metadata["feature_columns"]
            self.assertTrue(Path(metadata["client_rows_overview_path"]).exists())
            self.assertTrue(Path(metadata["client_fraud_overview_path"]).exists())

            seen_columns: list[list[str]] = []
            for client_entry in metadata["clients"]:
                train_frame = pd.read_csv(client_entry["train_path"])
                test_frame = pd.read_csv(client_entry["test_path"])
                self.assertEqual(train_frame.columns.tolist(), test_frame.columns.tolist())
                self.assertEqual(train_frame.columns.tolist(), feature_columns + [label_column])
                seen_columns.append(train_frame.columns.tolist())
            self.assertEqual(len(seen_columns), 5)

    def test_preparation_drops_sparse_rows_before_imputation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config = build_test_config(tmp_path, dataset_name="ieee_cis")
            transaction_path = config.paths.raw_transaction_path
            transaction_frame = pd.read_csv(transaction_path)
            transaction_frame = pd.concat(
                [
                    transaction_frame,
                    pd.DataFrame(
                        [
                            {
                                "TransactionID": 999999,
                                "isFraud": 1,
                                "card4": "visa",
                                "card6": "credit",
                                "TransactionAmt": float("nan"),
                                "dist1": float("nan"),
                                "V1": float("nan"),
                                "ProductCD": None,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
            transaction_frame.to_csv(transaction_path, index=False)

            prepared_root = run_preparation(config)
            metadata = read_json(prepared_root / "metadata.json")
            processed_rows = sum(int(client_entry["train_rows"]) + int(client_entry["test_rows"]) for client_entry in metadata["clients"])
            self.assertEqual(processed_rows, 60)

    def test_preparation_supports_additional_dataset_adapters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            for dataset_name in ["baf", "paysim", "credit_card_fraud", "elliptic"]:
                with self.subTest(dataset_name=dataset_name):
                    config = build_test_config(tmp_path / dataset_name, dataset_name=dataset_name)
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", pd.errors.PerformanceWarning)
                        prepared_root = run_preparation(config)
                    metadata = read_json(prepared_root / "metadata.json")
                    self.assertEqual(metadata["dataset_name"], dataset_name)
                    self.assertGreaterEqual(len(metadata["clients"]), 1)
                    self.assertTrue(Path(metadata["client_rows_overview_path"]).exists())
                    self.assertTrue(Path(metadata["client_fraud_overview_path"]).exists())
                    for client_entry in metadata["clients"]:
                        train_frame = pd.read_csv(client_entry["train_path"])
                        test_frame = pd.read_csv(client_entry["test_path"])
                        self.assertEqual(train_frame.columns.tolist(), test_frame.columns.tolist())
                        self.assertEqual(train_frame.columns.tolist(), metadata["feature_columns"] + [metadata["label_column"]])


if __name__ == "__main__":
    unittest.main()
