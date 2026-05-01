from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.pipeline.common import write_json

try:
    import pandas as pd
    from src.cli import build_parser
    from src.pipeline.stage2 import rerun_stage2_visualizations

    PIPELINE_IMPORT_ERROR = None
except ModuleNotFoundError as error:
    pd = None
    build_parser = None
    rerun_stage2_visualizations = None
    PIPELINE_IMPORT_ERROR = error


@unittest.skipIf(PIPELINE_IMPORT_ERROR is not None, f"pipeline dependencies unavailable: {PIPELINE_IMPORT_ERROR}")
class Stage2PipelineTest(unittest.TestCase):
    def test_stage2_visualize_command_accepts_optional_run(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            ["--config", "src/config_files/config_ieee_cis.yaml", "stage2-visualize", "--stage2-run", "stage2_20260420_120000"]
        )

        self.assertEqual(args.command, "stage2-visualize")
        self.assertEqual(args.stage2_run, "stage2_20260420_120000")

    def test_rerun_stage2_visualizations_recreates_plot_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stage2_root = Path(directory) / "stage2"
            run_dir = stage2_root / "stage2_20260420_120000"
            metrics_summary_path = run_dir / "classifier_metrics_summary.csv"
            pd.DataFrame(
                [
                    {
                        "model": "xgboost",
                        "synthetic_source": "collafuse",
                        "ratio": 0.0,
                        "f1_mean": 0.61,
                        "precision_mean": 0.58,
                        "recall_mean": 0.65,
                        "roc_auc_mean": 0.79,
                        "average_precision_mean": 0.72,
                    },
                    {
                        "model": "xgboost",
                        "synthetic_source": "collafuse",
                        "ratio": 1.0,
                        "f1_mean": 0.68,
                        "precision_mean": 0.64,
                        "recall_mean": 0.73,
                        "roc_auc_mean": 0.83,
                        "average_precision_mean": 0.77,
                    },
                ]
            ).to_csv(metrics_summary_path, index=False)
            write_json(
                run_dir / "run_manifest.json",
                {
                    "run_id": "stage2_20260420_120000",
                    "metrics_summary_path": str(metrics_summary_path),
                },
            )

            config = SimpleNamespace(paths=SimpleNamespace(stage2_root=stage2_root))

            resolved_run_dir = rerun_stage2_visualizations(config, stage2_run="stage2_20260420_120000")

            self.assertEqual(resolved_run_dir, run_dir.resolve())
            self.assertTrue((run_dir / "plots" / "xgboost_f1.png").exists())
            self.assertTrue((run_dir / "plots" / "xgboost_precision.png").exists())
            self.assertTrue((run_dir / "plots" / "xgboost_recall.png").exists())
            self.assertTrue((run_dir / "plots" / "xgboost_roc_auc.png").exists())
            self.assertTrue((run_dir / "plots" / "xgboost_average_precision.png").exists())


if __name__ == "__main__":
    unittest.main()
