from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.collafuse.generative_baselines import (
    get_enabled_pool_sources,
    get_enabled_ratio_sources,
    get_enabled_real_only_sources,
    normalize_stage1_synthetic_paths,
)
from tests.helpers import build_test_config


class BaselineConfigurationTest(unittest.TestCase):
    def test_enabled_source_selectors_cover_requested_baseline_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = build_test_config(Path(directory))

            self.assertEqual(
                get_enabled_real_only_sources(config),
                ["real_only_unweighted", "real_only_weighted"],
            )
            self.assertEqual(
                get_enabled_ratio_sources(config),
                [
                    "random_oversampling",
                    "smote",
                    "adasyn",
                    "collafuse",
                    "ctgan",
                    "tabddpm",
                    "local_only_ddpm",
                    "centralized_ddpm",
                ],
            )
            self.assertEqual(
                get_enabled_pool_sources(config),
                ["collafuse", "ctgan", "tabddpm", "local_only_ddpm", "centralized_ddpm"],
            )

    def test_stage1_manifest_normalization_supports_legacy_and_nested_layouts(self) -> None:
        legacy_manifest = {
            "synthetic_paths": {
                "CLIENT_0": "artifacts/stage1/legacy_client_0.csv",
                "CLIENT_1": "artifacts/stage1/legacy_client_1.csv",
            }
        }
        nested_manifest = {
            "synthetic_paths": {
                "collafuse": {
                    "CLIENT_0": "artifacts/stage1/collafuse_client_0.csv",
                },
                "ctgan": {
                    "CLIENT_0": "artifacts/stage1/ctgan_client_0.csv",
                },
            }
        }

        self.assertEqual(
            normalize_stage1_synthetic_paths(legacy_manifest),
            {
                "collafuse": {
                    "CLIENT_0": "artifacts/stage1/legacy_client_0.csv",
                    "CLIENT_1": "artifacts/stage1/legacy_client_1.csv",
                }
            },
        )
        self.assertEqual(
            normalize_stage1_synthetic_paths(nested_manifest),
            nested_manifest["synthetic_paths"],
        )


if __name__ == "__main__":
    unittest.main()
