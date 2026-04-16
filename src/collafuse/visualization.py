from __future__ import annotations

import logging
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_TEMP_CACHE_ROOT = Path(tempfile.gettempdir()) / "collafuse-plot-cache"
_TEMP_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_TEMP_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_TEMP_CACHE_ROOT / "xdg"))

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import to_rgb

from src.pipeline.common import ensure_directory

LOGGER = logging.getLogger("collafuse")

try:
    import scienceplots  # noqa: F401

    SCIENCEPLOTS_AVAILABLE = True
except ImportError:
    SCIENCEPLOTS_AVAILABLE = False


# Canva "Wrap of Colors" anchor palette:
# Cinnabar   #e7625f
# Freesia    #e2b850
# Blue Grotto #1f6cb0
# Rose Quartz #e24c60
#
# Greens and oranges below are derived support tones chosen to stay visually
# compatible with those anchors while giving each baseline family a stable color
# identity across Stage 1 and Stage 2.
SOURCE_DISPLAY_NAMES = {
    "real_fraud": "Real Fraud",
    "real_only_unweighted": "Real Only Unweighted",
    "real_only_weighted": "Real Only Weighted",
    "random_oversampling": "Random Oversampling",
    "smote": "SMOTE",
    "adasyn": "ADASYN",
    "collafuse": "CollaFuse",
    "ctgan": "CTGAN",
    "tabddpm": "TabDDPM",
    "local_only_ddpm": "Local Only DDPM",
    "centralized_ddpm": "Centralized DDPM",
}

SOURCE_ORDER = [
    "real_fraud",
    "real_only_unweighted",
    "real_only_weighted",
    "random_oversampling",
    "smote",
    "adasyn",
    "collafuse",
    "ctgan",
    "tabddpm",
    "local_only_ddpm",
    "centralized_ddpm",
]

SOURCE_COLORS = {
    "real_fraud": "#202124",
    "real_only_unweighted": "#3b3f45",
    "real_only_weighted": "#747b83",
    "random_oversampling": "#b8bec5",
    "smote": "#1f6cb0",
    "adasyn": "#6f9fcf",
    "collafuse": "#e7625f",
    "ctgan": "#5f9471",
    "tabddpm": "#86b29a",
    "local_only_ddpm": "#e69557",
    "centralized_ddpm": "#f0b16f",
}

SOURCE_MARKERS = {
    "real_fraud": "X",
    "real_only_unweighted": "s",
    "real_only_weighted": "D",
    "random_oversampling": "P",
    "smote": "^",
    "adasyn": "v",
    "collafuse": "o",
    "ctgan": "h",
    "tabddpm": "8",
    "local_only_ddpm": ">",
    "centralized_ddpm": "<",
}

SOURCE_LINESTYLES = {
    "real_only_unweighted": "-",
    "real_only_weighted": "--",
    "random_oversampling": ":",
    "smote": "-",
    "adasyn": "--",
    "collafuse": "-",
    "ctgan": "-",
    "tabddpm": "--",
    "local_only_ddpm": "-",
    "centralized_ddpm": "--",
}

COLLAFUSE_TRAINING_COLORS = {
    "client_loss": "#e7625f",
    "cloud_loss": "#b33e54",
    "l_norm": "#f09f63",
    "l_prior": "#f4c06a",
    "l_triplet": "#c95b6f",
    "client_noise_accuracy": "#e7625f",
    "cloud_noise_accuracy": "#b33e54",
}

PREPARATION_COLORS = {
    "train_rows": "#1f6cb0",
    "test_rows": "#86b5de",
    "train_fraud": "#e7625f",
    "test_fraud": "#f3a097",
}

DATASET_DISPLAY_NAMES = {
    "ieee_cis": "IEEE-CIS",
    "baf": "BAF",
    "paysim": "PaySim",
    "credit_card_fraud": "Credit Card Fraud",
    "elliptic": "Elliptic",
}

NEURIPS_LIKE_RC_PARAMS = {
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#1f1f1f",
    "axes.linewidth": 0.85,
    "axes.titleweight": "semibold",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.labelcolor": "#1f1f1f",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "legend.fontsize": 8.5,
    "legend.title_fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.color": "#2f2f2f",
    "ytick.color": "#2f2f2f",
    "grid.color": "#d8dde6",
    "grid.linewidth": 0.7,
    "grid.alpha": 0.55,
    "lines.linewidth": 1.9,
    "lines.markersize": 5.2,
    "savefig.facecolor": "#ffffff",
    "savefig.bbox": "tight",
}


@contextmanager
def _plot_style():
    if SCIENCEPLOTS_AVAILABLE:
        with plt.style.context(["science", "no-latex"]):
            with plt.rc_context(NEURIPS_LIKE_RC_PARAMS):
                yield
    else:
        with plt.rc_context(NEURIPS_LIKE_RC_PARAMS):
            yield


def _ordered_sources(source_values: list[str]) -> list[str]:
    ordered = [source for source in SOURCE_ORDER if source in source_values]
    remaining = sorted(source for source in source_values if source not in SOURCE_ORDER)
    return ordered + remaining


def _label_for_source(source: str) -> str:
    return SOURCE_DISPLAY_NAMES.get(source, source.replace("_", " ").title())


def _color_for_source(source: str) -> str:
    return SOURCE_COLORS.get(source, "#5c6b73")


def _title_case_label(raw_label: str) -> str:
    token_map = {
        "adasyn": "ADASYN",
        "auc": "AUC",
        "collafuse": "CollaFuse",
        "ctgan": "CTGAN",
        "ddpm": "DDPM",
        "f1": "F1",
        "mmd": "MMD",
        "pca": "PCA",
        "roc": "ROC",
        "smote": "SMOTE",
        "tabddpm": "TabDDPM",
        "tsne": "t-SNE",
    }
    tokens = raw_label.replace("_", " ").replace("-", " ").split()
    formatted_tokens = [token_map.get(token.lower(), token.capitalize()) for token in tokens]
    return " ".join(formatted_tokens)


def _contrast_text_color(color: str) -> str:
    red, green, blue = to_rgb(color)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "#ffffff" if luminance < 0.56 else "#1f1f1f"


def _annotate_segment_values(axis, x_positions, bottoms, heights, color: str, max_value: float) -> None:
    for x_pos, bottom, height in zip(x_positions, bottoms, heights, strict=False):
        if height <= 0:
            continue
        if height >= max_value * 0.08:
            axis.text(
                x_pos,
                bottom + (height / 2.0),
                f"{int(height):,}",
                ha="center",
                va="center",
                fontsize=8,
                color=_contrast_text_color(color),
            )
        else:
            axis.text(
                x_pos,
                bottom + height + (max_value * 0.02),
                f"{int(height):,}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#1f1f1f",
            )


def plot_source_embedding(embedding_frame: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    ensure_directory(output_path.parent)

    with _plot_style():
        figure, axis = plt.subplots(figsize=(10.5, 7))
        source_values = embedding_frame["synthetic_source"].dropna().unique().tolist()
        for source in _ordered_sources(source_values):
            group = embedding_frame.loc[embedding_frame["synthetic_source"] == source]
            axis.scatter(
                group["tsne_x"],
                group["tsne_y"],
                s=20 if source == "collafuse" else 14,
                alpha=0.78,
                label=_label_for_source(source),
                color=_color_for_source(source),
                marker=SOURCE_MARKERS.get(source, "o"),
                edgecolors="none",
            )
        axis.set_title("PCA + t-SNE Fraud Embedding", loc="left", pad=10)
        axis.set_xlabel("t-SNE 1")
        axis.set_ylabel("t-SNE 2")
        axis.grid(alpha=0.18, linewidth=0.7)
        axis.legend(frameon=False, ncol=2, fontsize=9)
        figure.tight_layout()
        figure.savefig(output_path, dpi=220)
        plt.close(figure)
    return output_path


def plot_mmd_boxplot(mmd_frame: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    ensure_directory(output_path.parent)

    ordered_sources = _ordered_sources(mmd_frame["synthetic_source"].dropna().unique().tolist())
    values = [mmd_frame.loc[mmd_frame["synthetic_source"] == source, "mmd"].to_list() for source in ordered_sources]
    with _plot_style():
        figure, axis = plt.subplots(figsize=(10.5, 5.5))
        boxplot = axis.boxplot(
            values,
            labels=[_label_for_source(source) for source in ordered_sources],
            patch_artist=True,
            medianprops={"color": "#1f1f1f", "linewidth": 1.4},
            whiskerprops={"color": "#5f6368", "linewidth": 1.1},
            capprops={"color": "#5f6368", "linewidth": 1.1},
        )
        for patch, source in zip(boxplot["boxes"], ordered_sources, strict=False):
            color = _color_for_source(source)
            patch.set_facecolor(color)
            patch.set_edgecolor(color)
            patch.set_alpha(0.86)

        axis.set_title("RQ1 Distributional Fidelity", loc="left", pad=10)
        axis.set_ylabel("MMD")
        axis.grid(axis="y", alpha=0.18, linewidth=0.7)
        axis.tick_params(axis="x", rotation=20)
        figure.tight_layout()
        figure.savefig(output_path, dpi=220)
        plt.close(figure)
    return output_path


def plot_metric_lines(summary_frame: pd.DataFrame, metric: str, output_path: str | Path, title: str) -> Path:
    output_path = Path(output_path)
    ensure_directory(output_path.parent)

    with _plot_style():
        figure, axis = plt.subplots(figsize=(10.5, 5.5))
        source_values = summary_frame["synthetic_source"].dropna().unique().tolist()
        for source in _ordered_sources(source_values):
            group = summary_frame.loc[summary_frame["synthetic_source"] == source]
            ordered = group.sort_values("ratio")
            color = _color_for_source(source)
            axis.plot(
                ordered["ratio"],
                ordered[f"{metric}_mean"],
                marker=SOURCE_MARKERS.get(source, "o"),
                markersize=5.5,
                linewidth=2.4 if source == "collafuse" else 1.8,
                linestyle=SOURCE_LINESTYLES.get(source, "-"),
                color=color,
                label=_label_for_source(source),
            )
        axis.set_title(_title_case_label(title), loc="left", pad=10)
        axis.set_xlabel("Synthetic Ratio")
        axis.set_ylabel(_title_case_label(metric))
        axis.grid(alpha=0.18, linewidth=0.7)
        axis.legend(frameon=False, ncol=2, fontsize=9)
        figure.tight_layout()
        figure.savefig(output_path, dpi=220)
        plt.close(figure)
    return output_path


def plot_collafuse_training_loss(history_frame: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    ensure_directory(output_path.parent)
    if history_frame.empty:
        with _plot_style():
            figure, axis = plt.subplots(figsize=(10.5, 5.5))
            axis.text(0.5, 0.5, "No training history available", ha="center", va="center")
            axis.axis("off")
            figure.tight_layout()
            figure.savefig(output_path, dpi=220)
            plt.close(figure)
        return output_path

    summary = (
        history_frame.groupby("epoch", dropna=False)[["client_loss", "cloud_loss", "l_norm", "l_prior", "l_triplet"]]
        .mean()
        .reset_index()
    )

    with _plot_style():
        figure, axis = plt.subplots(figsize=(10.5, 5.5))
        for client_id, client_group in history_frame.groupby("client_id"):
            axis.plot(
                client_group["epoch"],
                client_group["client_loss"],
                color=COLLAFUSE_TRAINING_COLORS["client_loss"],
                alpha=0.18,
                linewidth=1.1,
            )

        axis.plot(
            summary["epoch"],
            summary["client_loss"],
            color=COLLAFUSE_TRAINING_COLORS["client_loss"],
            linewidth=2.5,
            marker="o",
            label="Average Client Loss",
        )
        axis.plot(
            summary["epoch"],
            summary["cloud_loss"],
            color=COLLAFUSE_TRAINING_COLORS["cloud_loss"],
            linewidth=2.3,
            marker="s",
            label="Average Cloud Loss",
        )
        axis.plot(
            summary["epoch"],
            summary["l_norm"],
            color=COLLAFUSE_TRAINING_COLORS["l_norm"],
            linewidth=1.8,
            linestyle="--",
            label="L Norm",
        )
        axis.plot(
            summary["epoch"],
            summary["l_prior"],
            color=COLLAFUSE_TRAINING_COLORS["l_prior"],
            linewidth=1.8,
            linestyle=":",
            label="L Prior",
        )
        axis.plot(
            summary["epoch"],
            summary["l_triplet"],
            color=COLLAFUSE_TRAINING_COLORS["l_triplet"],
            linewidth=1.8,
            linestyle="-.",
            label="L Triplet",
        )
        axis.set_title("CollaFuse Training Loss", loc="left", pad=10)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.grid(alpha=0.18, linewidth=0.7)
        axis.legend(frameon=False, ncol=2, fontsize=9)
        figure.tight_layout()
        figure.savefig(output_path, dpi=220)
        plt.close(figure)
    return output_path


def plot_collafuse_training_accuracy(history_frame: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    ensure_directory(output_path.parent)
    if history_frame.empty:
        with _plot_style():
            figure, axis = plt.subplots(figsize=(10.5, 5.0))
            axis.text(0.5, 0.5, "No training history available", ha="center", va="center")
            axis.axis("off")
            figure.tight_layout()
            figure.savefig(output_path, dpi=220)
            plt.close(figure)
        return output_path

    summary = (
        history_frame.groupby("epoch", dropna=False)[["client_noise_accuracy", "cloud_noise_accuracy"]]
        .mean()
        .reset_index()
    )

    with _plot_style():
        figure, axis = plt.subplots(figsize=(10.5, 5.0))
        for client_id, client_group in history_frame.groupby("client_id"):
            axis.plot(
                client_group["epoch"],
                client_group["client_noise_accuracy"],
                color=COLLAFUSE_TRAINING_COLORS["client_noise_accuracy"],
                alpha=0.18,
                linewidth=1.1,
            )

        axis.plot(
            summary["epoch"],
            summary["client_noise_accuracy"],
            color=COLLAFUSE_TRAINING_COLORS["client_noise_accuracy"],
            linewidth=2.5,
            marker="o",
            label="Average Client Denoising Accuracy",
        )
        axis.plot(
            summary["epoch"],
            summary["cloud_noise_accuracy"],
            color=COLLAFUSE_TRAINING_COLORS["cloud_noise_accuracy"],
            linewidth=2.3,
            marker="s",
            label="Average Cloud Denoising Accuracy",
        )
        axis.set_title("CollaFuse Denoising Accuracy", loc="left", pad=10)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Noise Sign Accuracy")
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.18, linewidth=0.7)
        axis.legend(frameon=False, fontsize=9)
        figure.tight_layout()
        figure.savefig(output_path, dpi=220)
        plt.close(figure)
    return output_path


def _plot_preparation_stacked_bars(
    x_positions: list[int],
    x_tick_labels: list[str],
    bottom_values,
    top_values,
    bottom_label: str,
    top_label: str,
    bottom_color: str,
    top_color: str,
    y_label: str,
    output_path: Path,
) -> Path:
    max_total = float((bottom_values + top_values).max()) if len(bottom_values) and (bottom_values + top_values).max() > 0 else 1.0
    width = max(9.5, 2.15 * len(x_positions))

    with _plot_style():
        figure, axis = plt.subplots(figsize=(width, 5.8))
        axis.bar(x_positions, bottom_values, color=bottom_color, label=bottom_label, width=0.66)
        axis.bar(x_positions, top_values, bottom=bottom_values, color=top_color, label=top_label, width=0.66)
        _annotate_segment_values(axis, x_positions, [0.0] * len(x_positions), bottom_values, bottom_color, max_total)
        _annotate_segment_values(axis, x_positions, bottom_values, top_values, top_color, max_total)

        for x_pos, total in zip(x_positions, bottom_values + top_values, strict=False):
            axis.text(
                x_pos,
                total + (max_total * 0.03),
                f"{int(total):,}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#1f1f1f",
            )

        axis.set_ylabel(y_label)
        axis.set_xlabel("Client")
        axis.set_xticks(x_positions, x_tick_labels)
        axis.tick_params(axis="x", rotation=0)
        axis.grid(axis="y", alpha=0.18, linewidth=0.7)
        axis.legend(
            frameon=False,
            fontsize=8.5,
            ncol=2,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
        )
        figure.tight_layout(rect=[0, 0.08, 1, 1])
        figure.savefig(output_path, dpi=220)
        plt.close(figure)
    return output_path


def plot_preparation_client_overview(
    dataset_name: str,
    client_entries: list[dict[str, Any]],
    output_dir: str | Path,
    skipped_clients: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    ensure_directory(output_dir)
    rows_output_path = output_dir / "client_rows_overview.png"
    fraud_output_path = output_dir / "client_fraud_overview.png"

    client_frame = pd.DataFrame(client_entries).copy()
    if client_frame.empty:
        with _plot_style():
            for output_path in (rows_output_path, fraud_output_path):
                figure, axis = plt.subplots(figsize=(10.5, 5.0))
                axis.text(0.5, 0.5, "No prepared client metadata available", ha="center", va="center")
                axis.axis("off")
                figure.tight_layout()
                figure.savefig(output_path, dpi=220)
                plt.close(figure)
        return {"rows": str(rows_output_path), "fraud": str(fraud_output_path)}

    client_frame = client_frame.sort_values("client_id").reset_index(drop=True)
    dataset_label = DATASET_DISPLAY_NAMES.get(dataset_name, _title_case_label(dataset_name))
    x_tick_labels = [
        f"{client_id}\n{_title_case_label(str(client_label))}"
        for client_id, client_label in zip(client_frame["client_id"], client_frame["client_label"], strict=False)
    ]
    x_positions = list(range(len(client_frame)))
    train_rows = client_frame["train_rows"].astype(float).to_numpy()
    test_rows = client_frame["test_rows"].astype(float).to_numpy()
    train_fraud = client_frame["train_fraud"].astype(float).to_numpy()
    test_fraud = client_frame["test_fraud"].astype(float).to_numpy()

    _plot_preparation_stacked_bars(
        x_positions=x_positions,
        x_tick_labels=x_tick_labels,
        bottom_values=train_rows,
        top_values=test_rows,
        bottom_label="Train Rows",
        top_label="Test Rows",
        bottom_color=PREPARATION_COLORS["train_rows"],
        top_color=PREPARATION_COLORS["test_rows"],
        y_label=f"{dataset_label} Rows",
        output_path=rows_output_path,
    )
    _plot_preparation_stacked_bars(
        x_positions=x_positions,
        x_tick_labels=x_tick_labels,
        bottom_values=train_fraud,
        top_values=test_fraud,
        bottom_label="Train Fraud",
        top_label="Test Fraud",
        bottom_color=PREPARATION_COLORS["train_fraud"],
        top_color=PREPARATION_COLORS["test_fraud"],
        y_label=f"{dataset_label} Fraud Rows",
        output_path=fraud_output_path,
    )

    return {"rows": str(rows_output_path), "fraud": str(fraud_output_path)}
