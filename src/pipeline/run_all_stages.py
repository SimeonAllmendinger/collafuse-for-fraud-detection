from __future__ import annotations

from pathlib import Path

from src.config_files.configs import PaperPipelineConfig
from src.pipeline.common import ensure_directory, make_run_id, write_json
from src.pipeline.prepare import run_preparation
from src.pipeline.stage1 import run_stage1_generation
from src.pipeline.stage2 import run_stage2_evaluation


def run_all_stages_pipeline(config: PaperPipelineConfig, reuse_checkpoint: bool = False) -> Path:
    run_all_stages_id = make_run_id("run_all_stages")
    run_all_stages_dir = ensure_directory(config.paths.run_all_stages_root / run_all_stages_id)
    prepared_root = run_preparation(config)
    stage1_dir = run_stage1_generation(config, reuse_checkpoint=reuse_checkpoint)
    stage2_dir = run_stage2_evaluation(config, str(stage1_dir))
    write_json(
        run_all_stages_dir / "run_all_stages_manifest.json",
        {
            "run_all_stages_id": run_all_stages_id,
            "prepared_root": str(prepared_root),
            "stage1_dir": str(stage1_dir),
            "stage2_dir": str(stage2_dir),
        },
    )
    return run_all_stages_dir
