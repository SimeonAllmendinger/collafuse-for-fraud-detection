from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.config_files.config import load_pipeline_config


class ConfigLoadingTest(unittest.TestCase):
    def test_load_pipeline_config_supports_extends_and_deep_merge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            base_config = tmp_path / "base.yaml"
            child_config = tmp_path / "child.yaml"

            base_config.write_text(
                "\n".join(
                    [
                        "data:",
                        "  fill_strategy: median",
                        "  shuffle_seed: 11",
                        "collafuse:",
                        "  epochs: 12",
                        "classifiers:",
                        "  suite:",
                        "    - name: logistic_regression",
                        "      params:",
                        "        max_iter: 321",
                    ]
                ),
                encoding="utf-8",
            )
            child_config.write_text(
                "\n".join(
                    [
                        "extends: base.yaml",
                        "paths:",
                        "  prepared_root: artifacts/custom_prepared",
                        "data:",
                        "  dataset_name: baf",
                        "client_split:",
                        "  strategy: quantile",
                        "  source_column: month",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_pipeline_config(child_config)
            self.assertEqual(config.data.dataset_name, "baf")
            self.assertEqual(config.data.fill_strategy, "median")
            self.assertEqual(config.data.shuffle_seed, 11)
            self.assertEqual(config.collafuse.epochs, 12)
            self.assertEqual(config.classifiers.suite[0].params["max_iter"], 321)
            self.assertTrue(str(config.paths.prepared_root).endswith("artifacts/custom_prepared"))


if __name__ == "__main__":
    unittest.main()
