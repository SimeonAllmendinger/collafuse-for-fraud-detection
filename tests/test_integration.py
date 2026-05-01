from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from src.pipeline.common import read_json
from src.pipeline.prepare import run_preparation
from tests.helpers import build_test_config


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for stage1/stage2 integration tests")
class IntegrationTest(unittest.TestCase):
    def test_end_to_end_stage_pipeline_on_toy_data(self) -> None:
        from src.pipeline.stage1 import run_stage1_generation
        from src.pipeline.stage2 import run_stage2_evaluation

        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config = build_test_config(tmp_path)
            prepared_root = run_preparation(config)
            metadata = read_json(prepared_root / "metadata.json")
            self.assertEqual(len(metadata["clients"]), 5)

            stage1_dir = run_stage1_generation(config, reuse_checkpoint=False)
            stage1_manifest = read_json(stage1_dir / "run_manifest.json")
            self.assertTrue(Path(stage1_manifest["checkpoint_path"]).exists())
            self.assertTrue(Path(stage1_manifest["rq1_mmd_summary_path"]).exists())
            self.assertTrue(Path(stage1_manifest["rq1_mmd_boxplot_path"]).exists())
            self.assertTrue(Path(stage1_manifest["rq1_mmd_global_boxplot_path"]).exists())
            self.assertIn("collafuse", stage1_manifest["synthetic_paths"])
            for source_paths in stage1_manifest["synthetic_paths"].values():
                for synthetic_path in source_paths.values():
                    self.assertTrue(Path(synthetic_path).exists())

            stage2_dir = run_stage2_evaluation(config, stage1_run=str(stage1_dir))
            stage2_manifest = read_json(stage2_dir / "run_manifest.json")
            self.assertTrue(Path(stage2_manifest["metrics_raw_path"]).exists())
            self.assertTrue(Path(stage2_manifest["metrics_summary_path"]).exists())
            self.assertIn("fedavg_logistic_regression", Path(stage2_manifest["metrics_summary_path"]).read_text(encoding="utf-8"))

    def test_run_all_stages_writes_manifest(self) -> None:
        from src.pipeline.run_all_stages import run_all_stages_pipeline

        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config = build_test_config(tmp_path)
            run_all_stages_dir = run_all_stages_pipeline(config, reuse_checkpoint=False)
            manifest = read_json(run_all_stages_dir / "run_all_stages_manifest.json")
            self.assertTrue(Path(manifest["prepared_root"]).exists())
            self.assertTrue(Path(manifest["stage1_dir"]).exists())
            self.assertTrue(Path(manifest["stage2_dir"]).exists())


if __name__ == "__main__":
    unittest.main()
