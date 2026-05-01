from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from src.collafuse.visualization import (
    DATASET_DISPLAY_NAMES,
    SOURCE_DISPLAY_NAMES,
    SOURCE_ORDER,
    _plot_style,
)
from src.pipeline.common import ensure_directory, read_json


DEFAULT_ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
DEFAULT_OUTPUT_PATH = DEFAULT_ARTIFACTS_ROOT / "latest_stage1_rq1_mmd_all_datasets_boxplot.png"
DEFAULT_COMBINED_CSV_PATH = DEFAULT_ARTIFACTS_ROOT / "latest_stage1_rq1_mmd_all_datasets.csv"
DATASET_COLOR_ORDER = ["#1f6cb0", "#e7625f", "#5f9471", "#e69557", "#7b6ad6", "#6f9fcf"]


def _discover_stage1_roots(artifacts_root: Path) -> list[Path]:
    return sorted(path for path in artifacts_root.iterdir() if path.is_dir() and (path / "stage1").is_dir())


def _resolve_latest_stage1_run(stage1_root: Path) -> Path:
    candidates = sorted(path for path in stage1_root.iterdir() if path.is_dir() and (path / "run_manifest.json").exists())
    if not candidates:
        raise ValueError(f"No Stage 1 runs were found under: {stage1_root}")
    return candidates[-1]


def _dataset_key_for_root(dataset_root: Path) -> str:
    return dataset_root.name.replace("-", "_")


def _dataset_label(dataset_key: str) -> str:
    return DATASET_DISPLAY_NAMES.get(dataset_key, dataset_key.replace("_", " ").title())


def _source_label(source_name: str) -> str:
    return SOURCE_DISPLAY_NAMES.get(source_name, source_name.replace("_", " ").title())


def _ordered_sources(source_values: list[str]) -> list[str]:
    source_order = [source for source in SOURCE_ORDER if source != "collafuse"]
    if "collafuse" in source_values:
        source_order.append("collafuse")
    ordered = [source for source in source_order if source in source_values]
    remaining = sorted(source for source in source_values if source not in SOURCE_ORDER)
    return ordered + remaining


def _load_latest_stage1_mmd_frames(artifacts_root: Path, scope: str) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    frames: list[pd.DataFrame] = []
    manifests: list[dict[str, str]] = []

    for dataset_root in _discover_stage1_roots(artifacts_root):
        dataset_key = _dataset_key_for_root(dataset_root)
        latest_run_dir = _resolve_latest_stage1_run(dataset_root / "stage1")
        manifest = read_json(latest_run_dir / "run_manifest.json")
        raw_path = Path(manifest.get("rq1_mmd_raw_path", latest_run_dir / "rq1_mmd_raw.csv"))
        if not raw_path.exists():
            raise ValueError(f"Missing rq1_mmd_raw.csv for {dataset_root.name}: {raw_path}")

        frame = pd.read_csv(raw_path)
        if "comparison_scope" in frame.columns:
            frame = frame.loc[frame["comparison_scope"] == scope].copy()
        if frame.empty:
            continue

        frame["dataset_key"] = dataset_key
        frame["dataset_label"] = _dataset_label(dataset_key)
        frames.append(frame)
        manifests.append(
            {
                "dataset_key": dataset_key,
                "dataset_label": _dataset_label(dataset_key),
                "run_dir": str(latest_run_dir),
                "raw_path": str(raw_path),
            }
        )

    if not frames:
        raise ValueError(f"No Stage 1 MMD rows were found under {artifacts_root} for scope={scope!r}")

    return pd.concat(frames, ignore_index=True), manifests


def plot_combined_stage1_mmd_boxplot(frame: pd.DataFrame, output_path: Path) -> Path:
    output_path = Path(output_path)
    ensure_directory(output_path.parent)

    source_order = _ordered_sources(frame["synthetic_source"].dropna().unique().tolist())
    dataset_order = list(dict.fromkeys(frame["dataset_key"].tolist()))
    dataset_colors = {
        dataset_key: DATASET_COLOR_ORDER[index % len(DATASET_COLOR_ORDER)]
        for index, dataset_key in enumerate(dataset_order)
    }
    group_width = 0.68
    box_width = group_width / max(1, len(dataset_order))

    with _plot_style():
        figure, axis = plt.subplots(figsize=(max(12.0, len(source_order) * 1.35), 6.2))
        for source_index, source_name in enumerate(source_order):
            offsets = [
                (-group_width / 2.0) + (box_width / 2.0) + (dataset_index * box_width)
                for dataset_index in range(len(dataset_order))
            ]
            for dataset_index, dataset_key in enumerate(dataset_order):
                values = frame.loc[
                    (frame["synthetic_source"] == source_name) & (frame["dataset_key"] == dataset_key),
                    "mmd",
                ].dropna().tolist()
                if not values:
                    continue

                position = source_index + offsets[dataset_index]
                color = dataset_colors[dataset_key]
                boxplot = axis.boxplot(
                    [values],
                    positions=[position],
                    widths=box_width * 0.66,
                    patch_artist=True,
                    manage_ticks=False,
                    medianprops={"color": "#1f1f1f", "linewidth": 1.3},
                    whiskerprops={"color": "#5f6368", "linewidth": 1.0},
                    capprops={"color": "#5f6368", "linewidth": 1.0},
                )
                for patch in boxplot["boxes"]:
                    patch.set_facecolor(color)
                    patch.set_edgecolor(color)
                    patch.set_alpha(0.84)

        axis.set_xlabel("Models")
        axis.set_ylabel("MMD")
        axis.set_xticks(range(len(source_order)))
        axis.set_xticklabels([_source_label(source_name) for source_name in source_order], rotation=20, ha="right")
        axis.xaxis.label.set_size(16)
        axis.yaxis.label.set_size(18)
        axis.minorticks_off()
        axis.tick_params(axis="x", which="major", bottom=True, top=False, labelbottom=True, labelsize=16, length=4, width=0.85)
        axis.tick_params(axis="x", which="minor", bottom=False, top=False)
        axis.tick_params(axis="y", labelsize=16)
        axis.grid(axis="y", alpha=0.18, linewidth=0.7)

        legend_handles = [
            Patch(facecolor=dataset_colors[dataset_key], edgecolor=dataset_colors[dataset_key], alpha=0.84, label=_dataset_label(dataset_key))
            for dataset_key in dataset_order
        ]
        legend = axis.legend(handles=legend_handles, title="Dataset", frameon=False, ncol=min(3, len(legend_handles)))
        legend.get_title().set_fontsize(16)
        for text in legend.get_texts():
            text.set_fontsize(16)
        figure.tight_layout()
        figure.savefig(output_path, dpi=220)
        plt.close(figure)

    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot the latest Stage 1 RQ1 MMD results across all datasets")
    parser.add_argument("--artifacts-root", default=str(DEFAULT_ARTIFACTS_ROOT), help="Root directory containing per-dataset artifact folders")
    parser.add_argument("--scope", default="client", choices=["client", "global"], help="Which MMD scope to visualize from rq1_mmd_raw.csv")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Output PNG path")
    parser.add_argument(
        "--combined-csv-output",
        default=str(DEFAULT_COMBINED_CSV_PATH),
        help="Optional output path for the combined plotting dataframe",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    artifacts_root = Path(args.artifacts_root).resolve()
    combined_frame, run_records = _load_latest_stage1_mmd_frames(artifacts_root, scope=args.scope)

    combined_csv_output = Path(args.combined_csv_output).resolve()
    ensure_directory(combined_csv_output.parent)
    combined_frame.to_csv(combined_csv_output, index=False)

    output_path = plot_combined_stage1_mmd_boxplot(combined_frame, Path(args.output).resolve())
    print(output_path)
    for record in run_records:
        print(f"{record['dataset_label']}: {record['run_dir']}")


if __name__ == "__main__":
    main()
