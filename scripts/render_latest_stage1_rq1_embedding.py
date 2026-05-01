from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from src.collafuse.visualization import (
    DATASET_DISPLAY_NAMES,
    SOURCE_COLORS,
    SOURCE_DISPLAY_NAMES,
    SOURCE_MARKERS,
    SOURCE_ORDER,
    _plot_style,
    plot_source_embedding,
)
from src.pipeline.common import ensure_directory, read_json


DEFAULT_ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
DEFAULT_OUTPUT_PATH = DEFAULT_ARTIFACTS_ROOT / "latest_stage1_rq1_embedding_all_datasets.png"


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


def _source_color(source_name: str) -> str:
    return SOURCE_COLORS.get(source_name, "#5c6b73")


def _ordered_sources(source_values: list[str]) -> list[str]:
    source_order = [source for source in SOURCE_ORDER if source != "collafuse"]
    if "collafuse" in source_values:
        source_order.append("collafuse")
    ordered = [source for source in source_order if source in source_values]
    remaining = sorted(source for source in source_values if source not in SOURCE_ORDER)
    return ordered + remaining


def _load_latest_embedding_frames(artifacts_root: Path) -> tuple[list[tuple[str, pd.DataFrame]], list[dict[str, str]]]:
    frames: list[tuple[str, pd.DataFrame]] = []
    manifests: list[dict[str, str]] = []

    for dataset_root in _discover_stage1_roots(artifacts_root):
        dataset_key = _dataset_key_for_root(dataset_root)
        latest_run_dir = _resolve_latest_stage1_run(dataset_root / "stage1")
        manifest = read_json(latest_run_dir / "run_manifest.json")
        embedding_path = Path(manifest.get("rq1_embedding_path", latest_run_dir / "rq1_embedding.csv"))
        if not embedding_path.exists():
            raise ValueError(f"Missing rq1_embedding.csv for {dataset_root.name}: {embedding_path}")

        frame = pd.read_csv(embedding_path)
        required_columns = {"synthetic_source", "tsne_x", "tsne_y"}
        if not required_columns.issubset(frame.columns):
            raise ValueError(f"Embedding file is missing required columns for {dataset_root.name}: {embedding_path}")
        if frame.empty:
            continue

        frames.append((dataset_key, frame))
        manifests.append(
            {
                "dataset_key": dataset_key,
                "dataset_label": _dataset_label(dataset_key),
                "run_dir": str(latest_run_dir),
                "embedding_path": str(embedding_path),
                "plot_path": str(latest_run_dir / "rq1_embedding.png"),
            }
        )

    if not frames:
        raise ValueError(f"No Stage 1 embedding rows were found under {artifacts_root}")

    return frames, manifests


def plot_combined_embeddings(dataset_frames: list[tuple[str, pd.DataFrame]], output_path: Path) -> Path:
    output_path = Path(output_path)
    ensure_directory(output_path.parent)

    dataset_count = len(dataset_frames)
    ncols = min(3, dataset_count)
    nrows = math.ceil(dataset_count / ncols)
    figure_width = max(12.0, ncols * 4.6)
    figure_height = max(15.6, nrows * 5.9 + 1.0)
    all_sources = sorted({source for _, frame in dataset_frames for source in frame["synthetic_source"].dropna().unique().tolist()})
    source_order = _ordered_sources(all_sources)

    with _plot_style():
        figure, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figure_width, figure_height), squeeze=False)
        legend_handles: dict[str, object] = {}

        for axis, (dataset_key, frame) in zip(axes.flat, dataset_frames):
            for source_name in source_order:
                group = frame.loc[frame["synthetic_source"] == source_name]
                if group.empty:
                    continue
                scatter = axis.scatter(
                    group["tsne_x"],
                    group["tsne_y"],
                    s=18 if source_name == "collafuse" else 12,
                    alpha=0.78,
                    color=_source_color(source_name),
                    marker=SOURCE_MARKERS.get(source_name, "o"),
                    edgecolors="none",
                    label=_source_label(source_name),
                )
                legend_handles.setdefault(source_name, scatter)

            axis.set_title(_dataset_label(dataset_key), pad=8, fontsize=16)
            axis.grid(alpha=0.18, linewidth=0.7)
            axis.tick_params(axis="both", labelsize=16)

        for axis_index, axis in enumerate(axes.flat):
            if axis_index >= dataset_count:
                axis.axis("off")
                continue

            row_index = axis_index // ncols
            col_index = axis_index % ncols
            if row_index == nrows - 1:
                axis.set_xlabel("t-SNE 1")
                axis.xaxis.label.set_size(16)
            else:
                axis.set_xlabel("")
            if col_index == 0:
                axis.set_ylabel("t-SNE 2")
                axis.yaxis.label.set_size(16)
            else:
                axis.set_ylabel("")

        ordered_handles = [legend_handles[source_name] for source_name in source_order if source_name in legend_handles]
        ordered_labels = [_source_label(source_name) for source_name in source_order if source_name in legend_handles]
        figure.legend(
            ordered_handles,
            ordered_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=min(2, max(1, len(ordered_labels))),
            frameon=False,
            fontsize=16,
            markerscale=1.8,
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.9))
        figure.savefig(output_path, dpi=220)
        plt.close(figure)

    return output_path


def render_latest_dataset_embeddings(run_records: list[dict[str, str]]) -> list[Path]:
    rendered_paths: list[Path] = []
    for record in run_records:
        embedding_path = Path(record["embedding_path"])
        plot_path = Path(record["plot_path"])
        frame = pd.read_csv(embedding_path)
        if frame.empty:
            continue
        plot_source_embedding(frame, plot_path)
        rendered_paths.append(plot_path)
    return rendered_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render the latest Stage 1 PCA+t-SNE embeddings across all datasets")
    parser.add_argument("--artifacts-root", default=str(DEFAULT_ARTIFACTS_ROOT), help="Root directory containing per-dataset artifact folders")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Output PNG path")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    artifacts_root = Path(args.artifacts_root).resolve()
    dataset_frames, run_records = _load_latest_embedding_frames(artifacts_root)
    rendered_paths = render_latest_dataset_embeddings(run_records)
    output_path = plot_combined_embeddings(dataset_frames, Path(args.output).resolve())
    print(output_path)
    for record in run_records:
        print(f"{record['dataset_label']}: {record['run_dir']}")
    for rendered_path in rendered_paths:
        print(rendered_path)


if __name__ == "__main__":
    main()
