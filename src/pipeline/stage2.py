from __future__ import annotations

import logging
import warnings
from pathlib import Path

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from tqdm.auto import tqdm

from src.collafuse.augmentation import (
    compute_target_synthetic_count,
    generate_adasyn_samples,
    generate_random_oversampling_samples,
    generate_smote_samples,
    sample_from_pool,
)
from src.collafuse.classifiers import build_sample_weights, get_model_registry, predict_scores
from src.collafuse.generative_baselines import (
    get_enabled_pool_sources,
    get_enabled_ratio_sources,
    get_enabled_real_only_sources,
    normalize_stage1_synthetic_paths,
)
from src.collafuse.metrics import aggregate_metrics, compute_classification_metrics
from src.collafuse.visualization import plot_metric_lines
from src.config_files.configs import PaperPipelineConfig
from src.pipeline.common import ensure_directory, make_run_id, read_json, resolve_stage1_run, write_json

LOGGER = logging.getLogger("collafuse")


def _fit_model(model, X_train, y_train, sample_weight, context: str):
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        if sample_weight is None:
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train, sample_weight=sample_weight)

    if any(issubclass(warning.category, ConvergenceWarning) for warning in caught_warnings):
        LOGGER.warning(
            "Classifier convergence warning for %s. Consider increasing max_iter or using a different solver.",
            context,
        )
    return model


def run_stage2_evaluation(config: PaperPipelineConfig, stage1_run: str | None = None) -> Path:
    LOGGER.info("Starting Stage 2 evaluation")
    prepared_metadata = read_json(config.paths.prepared_root / "metadata.json")
    stage1_dir = resolve_stage1_run(config.paths.stage1_root, stage1_run)
    if stage1_run is None:
        LOGGER.info("No Stage 1 run was specified; using latest run at %s", stage1_dir)
    else:
        LOGGER.info("Using Stage 1 run at %s", stage1_dir)
    stage1_manifest = read_json(stage1_dir / "run_manifest.json")
    synthetic_pools = normalize_stage1_synthetic_paths(stage1_manifest)
    run_id = make_run_id("stage2")
    run_dir = ensure_directory(config.paths.stage2_root / run_id)
    model_registry = get_model_registry(config.classifiers.suite)
    label_column = prepared_metadata["label_column"]
    feature_columns = prepared_metadata["feature_columns"]
    enabled_real_sources = get_enabled_real_only_sources(config)
    enabled_ratio_sources = get_enabled_ratio_sources(config)
    enabled_pool_sources = set(get_enabled_pool_sources(config))
    available_ratio_sources: list[str] = []
    for source in enabled_ratio_sources:
        if source in {"random_oversampling", "smote", "adasyn"}:
            available_ratio_sources.append(source)
        elif source in synthetic_pools:
            available_ratio_sources.append(source)
        else:
            LOGGER.warning("Skipping Stage 2 source %s because no Stage 1 synthetic pool was found", source)
    num_clients = len(prepared_metadata["clients"])
    num_seeds = len(config.evaluation.classifier_seeds)
    num_models = len(model_registry)
    num_ratios = len(config.augmentation.ratios)
    total_model_fits = num_clients * num_seeds * num_models * (len(enabled_real_sources) + (num_ratios * len(available_ratio_sources)))
    LOGGER.info(
        "Stage 2 planned work: %s client(s) x %s seed(s) x %s model(s) with %s real baseline(s) and %s ratio source(s) over %s ratio(s) => %s total model fits",
        num_clients,
        num_seeds,
        num_models,
        len(enabled_real_sources),
        len(available_ratio_sources),
        num_ratios,
        total_model_fits,
    )

    raw_rows: list[dict[str, float | int | str]] = []

    with tqdm(total=total_model_fits, desc="Stage 2 evaluation", leave=True) as progress:
        for client_entry in prepared_metadata["clients"]:
            client_id = client_entry["client_id"]
            LOGGER.info("Evaluating client %s", client_id)
            train_frame = pd.read_csv(client_entry["train_path"])
            test_frame = pd.read_csv(client_entry["test_path"])
            pool_frames = {
                source_name: pd.read_csv(source_paths[client_id])
                for source_name, source_paths in synthetic_pools.items()
                if source_name in enabled_pool_sources and client_id in source_paths
            }

            X_test = test_frame[feature_columns].to_numpy(dtype="float32")
            y_test = test_frame[label_column].to_numpy(dtype="int64")
            X_base = train_frame[feature_columns].to_numpy(dtype="float32")
            y_base = train_frame[label_column].to_numpy(dtype="int64")
            base_sample_weight = build_sample_weights(y_base)

            for seed in config.evaluation.classifier_seeds:
                synthetic_data_by_ratio: dict[float, dict[str, tuple]] = {}

                for ratio in config.augmentation.ratios:
                    target_count = compute_target_synthetic_count(train_frame[label_column], ratio)
                    synthetic_sources: dict[str, pd.DataFrame] = {}
                    if "random_oversampling" in available_ratio_sources:
                        synthetic_sources["random_oversampling"] = generate_random_oversampling_samples(
                            train_df=train_frame,
                            label_column=label_column,
                            target_count=target_count,
                            random_state=seed,
                        )
                    if "smote" in available_ratio_sources:
                        synthetic_sources["smote"] = generate_smote_samples(
                            train_df=train_frame,
                            label_column=label_column,
                            target_count=target_count,
                            random_state=seed,
                            n_neighbors=config.augmentation.smote_k_neighbors,
                        )
                    if "adasyn" in available_ratio_sources:
                        synthetic_sources["adasyn"] = generate_adasyn_samples(
                            train_df=train_frame,
                            label_column=label_column,
                            target_count=target_count,
                            random_state=seed,
                            n_neighbors=config.augmentation.adasyn_n_neighbors,
                        )
                    for source_name, pool_frame in pool_frames.items():
                        if source_name in available_ratio_sources:
                            synthetic_sources[source_name] = sample_from_pool(pool_frame, target_count, seed)

                    prepared_sources: dict[str, tuple] = {}
                    for source_name, synthetic_frame in synthetic_sources.items():
                        augmented = pd.concat([train_frame, synthetic_frame], ignore_index=True)
                        X_train = augmented[feature_columns].to_numpy(dtype="float32")
                        y_train = augmented[label_column].to_numpy(dtype="int64")
                        prepared_sources[source_name] = (
                            X_train,
                            y_train,
                            build_sample_weights(y_train),
                        )
                    synthetic_data_by_ratio[ratio] = prepared_sources

                for model_name, factory in model_registry.items():
                    progress.set_postfix(client=client_id, model=model_name, seed=seed)
                    baseline_metrics_by_source: dict[str, dict[str, float]] = {}
                    if "real_only_unweighted" in enabled_real_sources:
                        baseline_model = factory(seed)
                        baseline_model = _fit_model(
                            baseline_model,
                            X_base,
                            y_base,
                            None,
                            context=f"{client_id} | {model_name} | real_only_unweighted | seed={seed}",
                        )
                        baseline_scores = predict_scores(baseline_model, X_test)
                        baseline_predictions = (baseline_scores >= 0.5).astype(int)
                        baseline_metrics_by_source["real_only_unweighted"] = compute_classification_metrics(y_test, baseline_predictions, baseline_scores)
                        progress.update(1)
                    if "real_only_weighted" in enabled_real_sources:
                        baseline_model = factory(seed)
                        baseline_model = _fit_model(
                            baseline_model,
                            X_base,
                            y_base,
                            base_sample_weight,
                            context=f"{client_id} | {model_name} | real_only_weighted | seed={seed}",
                        )
                        baseline_scores = predict_scores(baseline_model, X_test)
                        baseline_predictions = (baseline_scores >= 0.5).astype(int)
                        baseline_metrics_by_source["real_only_weighted"] = compute_classification_metrics(y_test, baseline_predictions, baseline_scores)
                        progress.update(1)

                    for ratio in config.augmentation.ratios:
                        for source_name, baseline_metrics in baseline_metrics_by_source.items():
                            raw_rows.append(
                                {
                                    "client_id": client_id,
                                    "model": model_name,
                                    "synthetic_source": source_name,
                                    "ratio": ratio,
                                    "seed": seed,
                                    **baseline_metrics,
                                }
                            )

                        for source_name, prepared_dataset in synthetic_data_by_ratio[ratio].items():
                            progress.set_postfix(client=client_id, model=model_name, seed=seed, ratio=ratio, source=source_name)
                            X_train, y_train, sample_weight = prepared_dataset
                            model = factory(seed)
                            model = _fit_model(
                                model,
                                X_train,
                                y_train,
                                sample_weight,
                                context=f"{client_id} | {model_name} | {source_name} | ratio={ratio} | seed={seed}",
                            )
                            y_scores = predict_scores(model, X_test)
                            y_predictions = (y_scores >= 0.5).astype(int)
                            metrics = compute_classification_metrics(y_test, y_predictions, y_scores)
                            raw_rows.append(
                                {
                                    "client_id": client_id,
                                    "model": model_name,
                                    "synthetic_source": source_name,
                                    "ratio": ratio,
                                    "seed": seed,
                                    **metrics,
                                }
                            )
                            progress.update(1)

    raw_frame = pd.DataFrame(raw_rows)
    raw_path = run_dir / "classifier_metrics_raw.csv"
    raw_frame.to_csv(raw_path, index=False)

    summary_by_model = aggregate_metrics(raw_frame, ["model", "synthetic_source", "ratio"])
    summary_by_model_path = run_dir / "classifier_metrics_summary.csv"
    summary_by_model.to_csv(summary_by_model_path, index=False)

    summary_by_client = aggregate_metrics(raw_frame, ["client_id", "model", "synthetic_source", "ratio"])
    summary_by_client_path = run_dir / "classifier_metrics_by_client.csv"
    summary_by_client.to_csv(summary_by_client_path, index=False)

    plots_dir = ensure_directory(run_dir / "plots")
    for model_name, model_frame in summary_by_model.groupby("model"):
        for metric in ["f1", "precision", "recall", "roc_auc", "average_precision"]:
            plot_metric_lines(
                model_frame,
                metric=metric,
                output_path=plots_dir / f"{model_name}_{metric}.png",
                title=f"{model_name} {metric.replace('_', ' ').title()}",
            )

    manifest = {
        "run_id": run_id,
        "stage1_run": str(stage1_dir),
        "metrics_raw_path": str(raw_path),
        "metrics_summary_path": str(summary_by_model_path),
        "metrics_by_client_path": str(summary_by_client_path),
        "plots_dir": str(plots_dir),
    }
    write_json(run_dir / "run_manifest.json", manifest)
    LOGGER.info("Stage 2 manifest saved to %s", run_dir / "run_manifest.json")
    LOGGER.info("Stage 2 complete")
    return run_dir
