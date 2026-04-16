from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from src.collafuse.augmentation import (
    generate_adasyn_samples,
    generate_random_oversampling_samples,
    generate_smote_samples,
)
from src.collafuse.engine import CollaFuseExperiment
from src.collafuse.generative_baselines import (
    generate_stage1_baseline_pools,
    get_enabled_rq1_sources,
)
from src.collafuse.metrics import build_pca_tsne_embedding, compute_mmd_rows
from src.collafuse.visualization import (
    plot_collafuse_training_accuracy,
    plot_collafuse_training_loss,
    plot_mmd_boxplot,
    plot_source_embedding,
)
from src.config_files.configs import PaperPipelineConfig
from src.pipeline.common import ensure_directory, make_run_id, read_json, write_json

LOGGER = logging.getLogger("collafuse")


def _attach_embedding_metadata(sample: pd.DataFrame, client_id: str, synthetic_source: str) -> pd.DataFrame:
    metadata = pd.DataFrame(
        {
            "client_id": [client_id] * len(sample),
            "synthetic_source": [synthetic_source] * len(sample),
        },
        index=sample.index,
    )
    return pd.concat([sample.copy(), metadata], axis=1)


def run_stage1_generation(config: PaperPipelineConfig, reuse_checkpoint: bool = False) -> Path:
    LOGGER.info("Starting Stage 1 generation")
    prepared_metadata = read_json(config.paths.prepared_root / "metadata.json")
    run_id = make_run_id("stage1")
    run_dir = ensure_directory(config.paths.stage1_root / run_id)
    LOGGER.info("Stage 1 run directory: %s", run_dir)
    LOGGER.info("Loaded prepared metadata for %s client(s)", len(prepared_metadata["clients"]))
    experiment = CollaFuseExperiment(config, prepared_metadata)

    LOGGER.info("Phase 1/4: train CollaFuse")
    checkpoint_path, training_history = experiment.train(run_dir, reuse_checkpoint=reuse_checkpoint)
    LOGGER.info("CollaFuse checkpoint ready at %s", checkpoint_path)
    collafuse_training_loss_path = run_dir / "collafuse_training_loss.png"
    collafuse_training_accuracy_path = run_dir / "collafuse_training_accuracy.png"
    plot_collafuse_training_loss(training_history, collafuse_training_loss_path)
    plot_collafuse_training_accuracy(training_history, collafuse_training_accuracy_path)

    LOGGER.info("Phase 2/4: sample CollaFuse synthetic fraud records")
    collafuse_paths = experiment.sample_all_clients(run_dir)
    synthetic_paths: dict[str, dict[str, str]] = {"collafuse": collafuse_paths}

    LOGGER.info("Phase 3/4: train additional generator baselines")
    baseline_synthetic_paths, baseline_model_paths = generate_stage1_baseline_pools(config, prepared_metadata, run_dir)
    synthetic_paths.update(baseline_synthetic_paths)

    label_column = prepared_metadata["label_column"]
    feature_columns = prepared_metadata["feature_columns"]
    mmd_rows: list[dict[str, Any]] = []
    embedding_frames: list[pd.DataFrame] = []

    LOGGER.info("Phase 4/4: build RQ1 artifacts and baseline samples")
    enabled_rq1_sources = set(get_enabled_rq1_sources(config))
    for client_entry in tqdm(prepared_metadata["clients"], desc="Stage 1 RQ1 artifacts", leave=True):
        client_id = client_entry["client_id"]
        LOGGER.info("Preparing RQ1 artifacts for %s", client_id)
        train_frame = pd.read_csv(client_entry["train_path"])
        real_fraud = train_frame.loc[train_frame[label_column] == 1].reset_index(drop=True)
        collafuse_frame = pd.read_csv(synthetic_paths["collafuse"][client_id])
        target_count = len(collafuse_frame)
        source_frames: dict[str, pd.DataFrame] = {}

        for source_name in enabled_rq1_sources:
            if source_name in synthetic_paths and client_id in synthetic_paths[source_name]:
                source_frames[source_name] = pd.read_csv(synthetic_paths[source_name][client_id])

        if "random_oversampling" in enabled_rq1_sources:
            source_frames["random_oversampling"] = generate_random_oversampling_samples(
                train_df=train_frame,
                label_column=label_column,
                target_count=target_count,
                random_state=config.sampling.random_seed,
                progress_desc=f"RandomOverSampling {client_id}",
            )
        if "smote" in enabled_rq1_sources:
            source_frames["smote"] = generate_smote_samples(
                train_df=train_frame,
                label_column=label_column,
                target_count=target_count,
                random_state=config.sampling.random_seed,
                n_neighbors=config.augmentation.smote_k_neighbors,
                progress_desc=f"SMOTE {client_id}",
            )
        if "adasyn" in enabled_rq1_sources:
            source_frames["adasyn"] = generate_adasyn_samples(
                train_df=train_frame,
                label_column=label_column,
                target_count=target_count,
                random_state=config.sampling.random_seed,
                n_neighbors=config.augmentation.adasyn_n_neighbors,
                progress_desc=f"ADASYN {client_id}",
            )

        for source_name, source_frame in source_frames.items():
            mmd_rows.extend(
                compute_mmd_rows(
                    real_samples=real_fraud[feature_columns].to_numpy(),
                    generated_samples=source_frame[feature_columns].to_numpy(),
                    seeds=config.evaluation.mmd_seeds,
                    batch_size=config.evaluation.mmd_batch_size,
                    source=source_name,
                    client_id=client_id,
                )
            )
            sample_size = min(300, len(source_frame))
            if sample_size:
                sample = source_frame.sample(
                    n=sample_size,
                    replace=len(source_frame) < sample_size,
                    random_state=config.evaluation.tsne_random_state
                )
                sample = _attach_embedding_metadata(sample, client_id=client_id, synthetic_source=source_name)
                embedding_frames.append(sample)

        real_sample_size = min(300, len(real_fraud))
        if real_sample_size:
            real_sample = real_fraud.sample(
                n=real_sample_size,
                replace=len(real_fraud) < real_sample_size,
                random_state=config.evaluation.tsne_random_state,
            )
            real_sample = _attach_embedding_metadata(real_sample, client_id=client_id, synthetic_source="real_fraud")
            embedding_frames.append(real_sample)

    mmd_frame = pd.DataFrame(mmd_rows)
    mmd_raw_path = run_dir / "rq1_mmd_raw.csv"
    mmd_frame.to_csv(mmd_raw_path, index=False)
    mmd_summary = (
        mmd_frame.groupby(["synthetic_source", "client_id"], dropna=False)["mmd"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "mmd_mean", "std": "mmd_std"})
    )
    mmd_summary_path = run_dir / "rq1_mmd_summary.csv"
    mmd_summary.to_csv(mmd_summary_path, index=False)

    embedding_frame = pd.concat(embedding_frames, ignore_index=True) if embedding_frames else pd.DataFrame()
    if not embedding_frame.empty:
        embedded = build_pca_tsne_embedding(
            frame=embedding_frame,
            feature_columns=feature_columns,
            pca_components=config.evaluation.pca_components,
            perplexity=config.evaluation.tsne_perplexity,
            learning_rate=config.evaluation.tsne_learning_rate,
            random_state=config.evaluation.tsne_random_state,
        )
        embedded_path = run_dir / "rq1_embedding.csv"
        embedded.to_csv(embedded_path, index=False)
        plot_source_embedding(embedded, run_dir / "rq1_embedding.png")
    else:
        embedded_path = run_dir / "rq1_embedding.csv"
        pd.DataFrame().to_csv(embedded_path, index=False)

    plot_mmd_boxplot(mmd_frame, run_dir / "rq1_mmd_boxplot.png")

    manifest = {
        "run_id": run_id,
        "prepared_root": str(config.paths.prepared_root),
        "checkpoint_path": str(checkpoint_path),
        "synthetic_paths": synthetic_paths,
        "baseline_model_paths": baseline_model_paths,
        "rq1_mmd_raw_path": str(mmd_raw_path),
        "rq1_mmd_summary_path": str(mmd_summary_path),
        "rq1_embedding_path": str(embedded_path),
        "training_history_path": str(run_dir / "training_history.csv"),
        "collafuse_training_loss_plot_path": str(collafuse_training_loss_path),
        "collafuse_training_accuracy_plot_path": str(collafuse_training_accuracy_path),
    }
    write_json(run_dir / "run_manifest.json", manifest)
    LOGGER.info("Stage 1 manifest saved to %s", run_dir / "run_manifest.json")
    LOGGER.info("Stage 1 complete")
    return run_dir
