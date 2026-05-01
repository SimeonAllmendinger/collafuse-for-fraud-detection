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
from src.collafuse.classifiers import build_sample_weights, get_model_registry, is_federated_classifier, predict_scores
from src.collafuse.generative_baselines import (
    get_enabled_pool_sources,
    get_enabled_ratio_sources,
    get_enabled_real_only_sources,
    normalize_stage1_synthetic_paths,
)
from src.collafuse.metrics import aggregate_metrics, compute_classification_metrics
from src.collafuse.visualization import plot_metric_lines
from src.config_files.configs import PaperPipelineConfig
from src.pipeline.common import (
    ensure_directory,
    make_run_id,
    read_json,
    resolve_stage1_run,
    resolve_stage2_run,
    serialize_config,
    write_json,
)

LOGGER = logging.getLogger("collafuse")
STAGE2_PLOT_METRICS = ("f1", "precision", "recall", "roc_auc", "average_precision")


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


def _evaluate_classifier(model, X_test, y_test) -> dict[str, float]:
    y_scores = predict_scores(model, X_test)
    y_predictions = (y_scores >= 0.5).astype(int)
    return compute_classification_metrics(y_test, y_predictions, y_scores)


def _append_metric_row(
    raw_rows: list[dict[str, float | int | str]],
    *,
    client_id: str,
    model_name: str,
    source_name: str,
    ratio: float,
    seed: int,
    metrics: dict[str, float],
) -> None:
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


def _prepare_synthetic_datasets(
    *,
    train_frame: pd.DataFrame,
    label_column: str,
    feature_columns: list[str],
    available_ratio_sources: list[str],
    pool_frames: dict[str, pd.DataFrame],
    config: PaperPipelineConfig,
    seed: int,
) -> dict[float, dict[str, tuple]]:
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

    return synthetic_data_by_ratio


def _fit_federated_model(model, client_datasets: list[dict[str, object]], context: str):
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit_federated(client_datasets)

    if any(issubclass(warning.category, ConvergenceWarning) for warning in caught_warnings):
        LOGGER.warning(
            "Classifier convergence warning for %s. Consider increasing max_iter or using a different solver.",
            context,
        )
    return model


def _render_stage2_plots(summary_by_model: pd.DataFrame, run_dir: Path) -> Path:
    plots_dir = ensure_directory(run_dir / "plots")
    for model_name, model_frame in summary_by_model.groupby("model"):
        for metric in STAGE2_PLOT_METRICS:
            plot_metric_lines(
                model_frame,
                metric=metric,
                output_path=plots_dir / f"{model_name}_{metric}.png",
                title=f"{model_name} {metric.replace('_', ' ').title()}",
            )
    return plots_dir


def rerun_stage2_visualizations(config: PaperPipelineConfig, stage2_run: str | None = None) -> Path:
    LOGGER.info("Re-rendering Stage 2 visualizations")
    run_dir = resolve_stage2_run(config.paths.stage2_root, stage2_run)
    if stage2_run is None:
        LOGGER.info("No Stage 2 run was specified; using latest run at %s", run_dir)
    else:
        LOGGER.info("Using Stage 2 run at %s", run_dir)

    manifest_path = run_dir / "run_manifest.json"
    manifest = read_json(manifest_path)
    metrics_summary_path = Path(manifest.get("metrics_summary_path", run_dir / "classifier_metrics_summary.csv"))
    if not metrics_summary_path.exists():
        raise ValueError(f"Stage 2 metrics summary does not exist: {metrics_summary_path}")

    summary_by_model = pd.read_csv(metrics_summary_path)
    plots_dir = _render_stage2_plots(summary_by_model, run_dir)
    manifest["metrics_summary_path"] = str(metrics_summary_path)
    manifest["plots_dir"] = str(plots_dir)
    write_json(manifest_path, manifest)
    LOGGER.info("Stage 2 plots saved to %s", plots_dir)
    return run_dir


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

    client_payloads: dict[str, dict[str, object]] = {}
    for client_entry in prepared_metadata["clients"]:
        client_id = client_entry["client_id"]
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
        client_payloads[client_id] = {
            "train_frame": train_frame,
            "pool_frames": pool_frames,
            "X_test": X_test,
            "y_test": y_test,
            "X_base": X_base,
            "y_base": y_base,
            "base_sample_weight": build_sample_weights(y_base),
        }

    num_clients = len(prepared_metadata["clients"])
    num_seeds = len(config.evaluation.classifier_seeds)
    local_model_names = [model_name for model_name in model_registry if not is_federated_classifier(model_name)]
    federated_model_names = [model_name for model_name in model_registry if is_federated_classifier(model_name)]
    num_ratios = len(config.augmentation.ratios)
    fit_units_per_seed = (
        len(local_model_names) * num_clients * (len(enabled_real_sources) + (num_ratios * len(available_ratio_sources)))
        + len(federated_model_names) * (len(enabled_real_sources) + (num_ratios * len(available_ratio_sources)))
    )
    total_model_fits = num_seeds * fit_units_per_seed
    LOGGER.info(
        "Stage 2 planned work: %s client(s), %s seed(s), %s local model(s), %s federated model(s), %s real baseline(s), and %s ratio source(s) over %s ratio(s) => %s total model fits",
        num_clients,
        num_seeds,
        len(local_model_names),
        len(federated_model_names),
        len(enabled_real_sources),
        len(available_ratio_sources),
        num_ratios,
        total_model_fits,
    )

    raw_rows: list[dict[str, float | int | str]] = []

    with tqdm(total=total_model_fits, desc="Stage 2 evaluation", leave=True) as progress:
        for seed in config.evaluation.classifier_seeds:
            prepared_by_client: dict[str, dict[str, object]] = {}
            for client_id, payload in client_payloads.items():
                LOGGER.info("Preparing Stage 2 datasets for %s at seed=%s", client_id, seed)
                prepared_by_client[client_id] = {
                    **payload,
                    "synthetic_data_by_ratio": _prepare_synthetic_datasets(
                        train_frame=payload["train_frame"],
                        label_column=label_column,
                        feature_columns=feature_columns,
                        available_ratio_sources=available_ratio_sources,
                        pool_frames=payload["pool_frames"],
                        config=config,
                        seed=seed,
                    ),
                }

            for model_name, factory in model_registry.items():
                if is_federated_classifier(model_name):
                    progress.set_postfix(model=model_name, seed=seed, scope="federated")
                    baseline_metrics_by_source: dict[str, dict[str, dict[str, float]]] = {}

                    if "real_only_unweighted" in enabled_real_sources:
                        client_datasets = [
                            {
                                "client_id": client_id,
                                "X": payload["X_base"],
                                "y": payload["y_base"],
                                "sample_weight": None,
                            }
                            for client_id, payload in prepared_by_client.items()
                        ]
                        federated_model = factory(seed)
                        federated_model = _fit_federated_model(
                            federated_model,
                            client_datasets,
                            context=f"{model_name} | real_only_unweighted | seed={seed}",
                        )
                        baseline_metrics_by_source["real_only_unweighted"] = {
                            client_id: _evaluate_classifier(federated_model, payload["X_test"], payload["y_test"])
                            for client_id, payload in prepared_by_client.items()
                        }
                        progress.update(1)

                    if "real_only_weighted" in enabled_real_sources:
                        client_datasets = [
                            {
                                "client_id": client_id,
                                "X": payload["X_base"],
                                "y": payload["y_base"],
                                "sample_weight": payload["base_sample_weight"],
                            }
                            for client_id, payload in prepared_by_client.items()
                        ]
                        federated_model = factory(seed)
                        federated_model = _fit_federated_model(
                            federated_model,
                            client_datasets,
                            context=f"{model_name} | real_only_weighted | seed={seed}",
                        )
                        baseline_metrics_by_source["real_only_weighted"] = {
                            client_id: _evaluate_classifier(federated_model, payload["X_test"], payload["y_test"])
                            for client_id, payload in prepared_by_client.items()
                        }
                        progress.update(1)

                    for ratio in config.augmentation.ratios:
                        for source_name, metrics_by_client in baseline_metrics_by_source.items():
                            for client_id, metrics in metrics_by_client.items():
                                _append_metric_row(
                                    raw_rows,
                                    client_id=client_id,
                                    model_name=model_name,
                                    source_name=source_name,
                                    ratio=ratio,
                                    seed=seed,
                                    metrics=metrics,
                                )

                        for source_name in available_ratio_sources:
                            progress.set_postfix(model=model_name, seed=seed, ratio=ratio, source=source_name, scope="federated")
                            client_datasets = []
                            evaluation_targets: list[tuple[str, dict[str, object]]] = []
                            for client_id, payload in prepared_by_client.items():
                                prepared_dataset = payload["synthetic_data_by_ratio"][ratio].get(source_name)
                                if prepared_dataset is None:
                                    continue
                                X_train, y_train, sample_weight = prepared_dataset
                                client_datasets.append(
                                    {
                                        "client_id": client_id,
                                        "X": X_train,
                                        "y": y_train,
                                        "sample_weight": sample_weight,
                                    }
                                )
                                evaluation_targets.append((client_id, payload))

                            if not client_datasets:
                                progress.update(1)
                                continue

                            federated_model = factory(seed)
                            federated_model = _fit_federated_model(
                                federated_model,
                                client_datasets,
                                context=f"{model_name} | {source_name} | ratio={ratio} | seed={seed}",
                            )
                            for client_id, payload in evaluation_targets:
                                metrics = _evaluate_classifier(federated_model, payload["X_test"], payload["y_test"])
                                _append_metric_row(
                                    raw_rows,
                                    client_id=client_id,
                                    model_name=model_name,
                                    source_name=source_name,
                                    ratio=ratio,
                                    seed=seed,
                                    metrics=metrics,
                                )
                            progress.update(1)
                    continue

                for client_id, payload in prepared_by_client.items():
                    LOGGER.info("Evaluating client %s with %s at seed=%s", client_id, model_name, seed)
                    X_test = payload["X_test"]
                    y_test = payload["y_test"]
                    X_base = payload["X_base"]
                    y_base = payload["y_base"]
                    base_sample_weight = payload["base_sample_weight"]
                    synthetic_data_by_ratio = payload["synthetic_data_by_ratio"]

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
                        baseline_metrics_by_source["real_only_unweighted"] = _evaluate_classifier(baseline_model, X_test, y_test)
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
                        baseline_metrics_by_source["real_only_weighted"] = _evaluate_classifier(baseline_model, X_test, y_test)
                        progress.update(1)

                    for ratio in config.augmentation.ratios:
                        for source_name, baseline_metrics in baseline_metrics_by_source.items():
                            _append_metric_row(
                                raw_rows,
                                client_id=client_id,
                                model_name=model_name,
                                source_name=source_name,
                                ratio=ratio,
                                seed=seed,
                                metrics=baseline_metrics,
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
                            metrics = _evaluate_classifier(model, X_test, y_test)
                            _append_metric_row(
                                raw_rows,
                                client_id=client_id,
                                model_name=model_name,
                                source_name=source_name,
                                ratio=ratio,
                                seed=seed,
                                metrics=metrics,
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

    plots_dir = _render_stage2_plots(summary_by_model, run_dir)

    manifest = {
        "run_id": run_id,
        "config": serialize_config(config),
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
