from __future__ import annotations

import argparse
import logging

from src.config_files.config import load_pipeline_config
from src.pipeline.prepare import run_preparation
from src.pipeline.run_all_stages import run_all_stages_pipeline
from src.pipeline.stage1 import run_stage1_generation
from src.pipeline.stage2 import run_stage2_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.cli", description="ICIS-style CollaFuse fraud pipeline")
    parser.add_argument("--config", required=True, help="Path to a dataset-specific pipeline config YAML")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare-data",
        help="Prepare the configured raw fraud dataset into standardized client splits"
    )
    prepare_parser.add_argument("--raw-transaction", help="Override the IEEE transaction CSV path")
    prepare_parser.add_argument("--raw-identity", help="Override the IEEE identity CSV path")
    prepare_parser.add_argument("--raw-main", help="Override the main raw table path for non-IEEE datasets")
    prepare_parser.add_argument("--raw-aux", help="Override the auxiliary raw table path for datasets that need one")
    prepare_parser.add_argument(
        "--raw-edge",
        help="Override the raw edge list path for graph-derived datasets such as Elliptic"
    )

    stage1_parser = subparsers.add_parser(
        "stage1-generate",
        help="Train CollaFuse and generate synthetic fraud samples"
    )
    stage1_parser.add_argument(
        "--reuse-checkpoint",
        action="store_true",
        help="Reuse an existing Stage 1 checkpoint when available"
    )

    stage2_parser = subparsers.add_parser("stage2-evaluate", help="Benchmark classifiers against Stage 1 outputs")
    stage2_parser.add_argument(
        "--stage1-run",
        help="Stage 1 run id or absolute path; defaults to the latest Stage 1 run"
    )

    run_parser = subparsers.add_parser(
        "run-all-stages",
        help="Execute preparation, Stage 1, and Stage 2 end-to-end"
    )
    run_parser.add_argument(
        "--reuse-checkpoint",
        action="store_true",
        help="Reuse an existing Stage 1 checkpoint when available"
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()
    config = load_pipeline_config(args.config)

    if args.command == "prepare-data":
        prepared_root = run_preparation(
            config,
            raw_transaction_path=args.raw_transaction,
            raw_identity_path=args.raw_identity,
            raw_main_path=args.raw_main,
            raw_aux_path=args.raw_aux,
            raw_edge_path=args.raw_edge,
        )
        print(prepared_root)
    elif args.command == "stage1-generate":
        run_dir = run_stage1_generation(config, reuse_checkpoint=args.reuse_checkpoint)
        print(run_dir)
    elif args.command == "stage2-evaluate":
        run_dir = run_stage2_evaluation(config, stage1_run=args.stage1_run)
        print(run_dir)
    elif args.command == "run-all-stages":
        run_all_stages_dir = run_all_stages_pipeline(config, reuse_checkpoint=args.reuse_checkpoint)
        print(run_all_stages_dir)


if __name__ == "__main__":
    main()
