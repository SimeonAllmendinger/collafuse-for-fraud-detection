from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.pipeline.common import resolve_stage1_run, resolve_stage2_run, write_json


class CommonPipelineTest(unittest.TestCase):
    def test_resolve_stage1_run_uses_latest_manifest_directory_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stage1_root = Path(directory) / "stage1"
            older_run = stage1_root / "stage1_20260415_101010"
            newer_run = stage1_root / "stage1_20260415_121212"
            write_json(older_run / "run_manifest.json", {"run_id": "older"})
            write_json(newer_run / "run_manifest.json", {"run_id": "newer"})

            resolved = resolve_stage1_run(stage1_root)

            self.assertEqual(resolved, newer_run.resolve())

    def test_resolve_stage1_run_accepts_explicit_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stage1_root = Path(directory) / "stage1"
            target_run = stage1_root / "stage1_20260415_101010"
            write_json(target_run / "run_manifest.json", {"run_id": "target"})

            resolved = resolve_stage1_run(stage1_root, "stage1_20260415_101010")

            self.assertEqual(resolved, target_run.resolve())

    def test_resolve_stage2_run_uses_latest_manifest_directory_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stage2_root = Path(directory) / "stage2"
            older_run = stage2_root / "stage2_20260415_101010"
            newer_run = stage2_root / "stage2_20260415_121212"
            write_json(older_run / "run_manifest.json", {"run_id": "older"})
            write_json(newer_run / "run_manifest.json", {"run_id": "newer"})

            resolved = resolve_stage2_run(stage2_root)

            self.assertEqual(resolved, newer_run.resolve())

    def test_resolve_stage2_run_accepts_explicit_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stage2_root = Path(directory) / "stage2"
            target_run = stage2_root / "stage2_20260415_101010"
            write_json(target_run / "run_manifest.json", {"run_id": "target"})

            resolved = resolve_stage2_run(stage2_root, "stage2_20260415_101010")

            self.assertEqual(resolved, target_run.resolve())


if __name__ == "__main__":
    unittest.main()
