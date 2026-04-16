from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from src.DDPM.DDPM import DDPM
from src.DDPM.learning_rate_schedulers import get_custom_cosine_schedule_with_warmup
from src.DDPM.losses import fraud_diffuse_loss
from src.DDPM.time_embedding import get_sinusoidal_embedding
from src.collafuse.dataset import TabularFraudDataset
from src.config_files.configs import BaselineSource, PaperPipelineConfig
from src.pipeline.common import ensure_directory, select_torch_device, set_random_seed

try:
    from ctgan import CTGAN

    CTGAN_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on optional package
    CTGAN = None
    CTGAN_AVAILABLE = False


LOGGER = logging.getLogger("collafuse")

REAL_ONLY_SOURCES: tuple[BaselineSource, ...] = (
    "real_only_unweighted",
    "real_only_weighted",
)
RATIO_GENERATION_SOURCES: tuple[BaselineSource, ...] = (
    "random_oversampling",
    "smote",
    "adasyn",
    "collafuse",
    "ctgan",
    "tabddpm",
    "local_only_ddpm",
    "centralized_ddpm",
)
POOL_BASED_SOURCES: tuple[BaselineSource, ...] = (
    "collafuse",
    "ctgan",
    "tabddpm",
    "local_only_ddpm",
    "centralized_ddpm",
)
RQ1_GENERATION_SOURCES: tuple[BaselineSource, ...] = (
    "random_oversampling",
    "smote",
    "adasyn",
    "collafuse",
    "ctgan",
    "tabddpm",
    "local_only_ddpm",
    "centralized_ddpm",
)


def get_enabled_sources(config: PaperPipelineConfig) -> set[str]:
    return set(config.baselines.enabled_sources)


def get_enabled_real_only_sources(config: PaperPipelineConfig) -> list[str]:
    enabled = get_enabled_sources(config)
    return [source for source in REAL_ONLY_SOURCES if source in enabled]


def get_enabled_ratio_sources(config: PaperPipelineConfig) -> list[str]:
    enabled = get_enabled_sources(config)
    return [source for source in RATIO_GENERATION_SOURCES if source in enabled]


def get_enabled_pool_sources(config: PaperPipelineConfig) -> list[str]:
    enabled = get_enabled_sources(config)
    return [source for source in POOL_BASED_SOURCES if source in enabled]


def get_enabled_rq1_sources(config: PaperPipelineConfig) -> list[str]:
    enabled = get_enabled_sources(config)
    return [source for source in RQ1_GENERATION_SOURCES if source in enabled]


def normalize_stage1_synthetic_paths(stage1_manifest: dict[str, Any]) -> dict[str, dict[str, str]]:
    raw_paths = stage1_manifest.get("synthetic_paths", {})
    if not raw_paths:
        return {}
    first_value = next(iter(raw_paths.values()))
    if isinstance(first_value, str):
        return {"collafuse": raw_paths}
    return raw_paths


def _ddpm_epochs(config: PaperPipelineConfig) -> int:
    return config.baselines.ddpm_epochs or config.collafuse.epochs


def _ddpm_batch_size(config: PaperPipelineConfig) -> int:
    return config.baselines.ddpm_batch_size or config.collafuse.batch_size


def _build_ddpm_model(config: PaperPipelineConfig, input_dim: int, device: torch.device) -> DDPM:
    return DDPM(
        device=device,
        input_dim=input_dim,
        T=config.collafuse.num_timesteps,
        beta_start=config.collafuse.beta_start,
        beta_end=config.collafuse.beta_end,
        time_embed_dim=config.collafuse.time_embed_dim,
    ).to(device)


def _sample_triplets(
    model: DDPM,
    fraud_pool: torch.Tensor,
    nonfraud_pool: torch.Tensor,
    nonfraud_mean: torch.Tensor,
    nonfraud_std: torch.Tensor,
    batch_size: int,
    feature_dim: int,
    device: torch.device,
    time_embed_dim: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    triplet_batch_size = min(batch_size, len(fraud_pool), len(nonfraud_pool))
    if triplet_batch_size == 0:
        return None, None, None

    fraud_indices = torch.randint(0, len(fraud_pool), (triplet_batch_size,), device=device)
    nonfraud_indices = torch.randint(0, len(nonfraud_pool), (triplet_batch_size,), device=device)
    positive = fraud_pool[fraud_indices]
    negative = nonfraud_pool[nonfraud_indices]

    z = torch.randn((triplet_batch_size, feature_dim), device=device)
    x_t = nonfraud_mean + nonfraud_std * z
    zero_steps = torch.zeros((triplet_batch_size,), dtype=torch.long, device=device)
    t_emb = get_sinusoidal_embedding(zero_steps, embedding_dim=time_embed_dim).to(device)
    noise_prediction = model.noise_predictor(x_t, t_emb)
    sqrt_alpha_bar_0 = model.beta_schedule.sqrt_alpha_bars[0]
    one_minus_alpha_bar_0 = model.beta_schedule.one_minus_alpha_bars[0]
    anchor = (x_t - one_minus_alpha_bar_0 * noise_prediction) / sqrt_alpha_bar_0
    return anchor, positive, negative


def _train_guided_ddpm(
    frame: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    config: PaperPipelineConfig,
    device: torch.device,
    progress_desc: str,
) -> DDPM:
    dataset = TabularFraudDataset(frame, label_column=label_column)
    dataloader = DataLoader(
        dataset,
        batch_size=_ddpm_batch_size(config),
        shuffle=True,
        num_workers=config.collafuse.num_workers,
    )
    model = _build_ddpm_model(config, len(feature_columns), device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.collafuse.learning_rate)
    scheduler = get_custom_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.collafuse.lr_warmup_steps,
        num_training_steps=max(1, len(dataloader) * _ddpm_epochs(config)),
    )

    fraud_pool = torch.tensor(
        frame.loc[frame[label_column] == 1, feature_columns].to_numpy(dtype="float32"),
        device=device,
    )
    nonfraud_pool = torch.tensor(
        frame.loc[frame[label_column] == 0, feature_columns].to_numpy(dtype="float32"),
        device=device,
    )
    nonfraud_mean = (
        nonfraud_pool.mean(dim=0, keepdim=True)
        if len(nonfraud_pool) > 0
        else torch.zeros((1, len(feature_columns)), device=device)
    )
    nonfraud_std = (
        nonfraud_pool.std(dim=0, keepdim=True, unbiased=False) + 1e-5
        if len(nonfraud_pool) > 0
        else torch.ones((1, len(feature_columns)), device=device)
    )

    epochs = _ddpm_epochs(config)
    with tqdm(range(1, epochs + 1), desc=progress_desc, leave=True) as epoch_progress:
        for _epoch in epoch_progress:
            loss_total = 0.0
            steps = 0
            for batch_features, _batch_labels in dataloader:
                x_0 = batch_features.to(device)
                batch_size = x_0.shape[0]
                noise = torch.randn_like(x_0)
                timesteps = torch.randint(0, config.collafuse.num_timesteps, (batch_size,), device=device, dtype=torch.long)
                sqrt_alpha = model.beta_schedule.sqrt_alpha_bars[timesteps].unsqueeze(1)
                sqrt_one_minus_alpha = model.beta_schedule.one_minus_alpha_bars[timesteps].unsqueeze(1)
                x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
                t_emb = get_sinusoidal_embedding(timesteps, embedding_dim=config.collafuse.time_embed_dim).to(device)
                predicted_noise = model.noise_predictor(x_t, t_emb)
                anchor, positive, negative = _sample_triplets(
                    model,
                    fraud_pool,
                    nonfraud_pool,
                    nonfraud_mean,
                    nonfraud_std,
                    batch_size,
                    len(feature_columns),
                    device,
                    config.collafuse.time_embed_dim,
                )
                loss, _components = fraud_diffuse_loss(
                    predicted_noise=predicted_noise,
                    true_noise=noise,
                    mu_nf=nonfraud_mean,
                    sigma_nf=nonfraud_std,
                    anchor=anchor,
                    positive=positive,
                    negative=negative,
                    w1=config.collafuse.w1,
                    w2=config.collafuse.w2,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                loss_total += float(loss.detach().item())
                steps += 1
            epoch_progress.set_postfix(loss=f"{loss_total / max(1, steps):.4f}")
    return model.eval()


def _train_standard_ddpm(
    fraud_frame: pd.DataFrame,
    feature_columns: list[str],
    config: PaperPipelineConfig,
    device: torch.device,
    progress_desc: str,
) -> DDPM:
    features = torch.tensor(fraud_frame[feature_columns].to_numpy(dtype="float32"), dtype=torch.float32)
    dataloader = DataLoader(
        TensorDataset(features),
        batch_size=_ddpm_batch_size(config),
        shuffle=True,
        num_workers=config.collafuse.num_workers,
    )
    model = _build_ddpm_model(config, len(feature_columns), device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.collafuse.learning_rate)
    scheduler = get_custom_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.collafuse.lr_warmup_steps,
        num_training_steps=max(1, len(dataloader) * _ddpm_epochs(config)),
    )

    epochs = _ddpm_epochs(config)
    with tqdm(range(1, epochs + 1), desc=progress_desc, leave=True) as epoch_progress:
        for _epoch in epoch_progress:
            loss_total = 0.0
            steps = 0
            for (batch_features,) in dataloader:
                x_0 = batch_features.to(device)
                batch_size = x_0.shape[0]
                noise = torch.randn_like(x_0)
                timesteps = torch.randint(0, config.collafuse.num_timesteps, (batch_size,), device=device, dtype=torch.long)
                sqrt_alpha = model.beta_schedule.sqrt_alpha_bars[timesteps].unsqueeze(1)
                sqrt_one_minus_alpha = model.beta_schedule.one_minus_alpha_bars[timesteps].unsqueeze(1)
                x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
                t_emb = get_sinusoidal_embedding(timesteps, embedding_dim=config.collafuse.time_embed_dim).to(device)
                predicted_noise = model.noise_predictor(x_t, t_emb)
                loss = F.mse_loss(predicted_noise, noise)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                loss_total += float(loss.detach().item())
                steps += 1
            epoch_progress.set_postfix(loss=f"{loss_total / max(1, steps):.4f}")
    return model.eval()


@torch.no_grad()
def _sample_ddpm_model(
    model: DDPM,
    feature_columns: list[str],
    label_column: str,
    n_samples: int,
    config: PaperPipelineConfig,
    device: torch.device,
    progress_desc: str,
) -> pd.DataFrame:
    if n_samples <= 0:
        return pd.DataFrame(columns=feature_columns + [label_column])

    outputs = []
    batch_size = config.sampling.sample_batch_size
    total_batches = list(range(0, n_samples, batch_size))
    with tqdm(total=len(total_batches), desc=progress_desc, leave=True) as progress:
        for start in total_batches:
            current_batch_size = min(batch_size, n_samples - start)
            samples = model.generate(
                shape=(current_batch_size, len(feature_columns)),
                device=device,
                num_of_inference_steps=config.sampling.n_inference_timesteps or config.collafuse.num_timesteps,
            )
            outputs.append(samples.detach().cpu())
            progress.update(1)

    frame = pd.DataFrame(torch.cat(outputs, dim=0).numpy(), columns=feature_columns)
    frame[label_column] = 1
    return frame


def _fit_ctgan(
    fraud_frame: pd.DataFrame,
    feature_columns: list[str],
    config: PaperPipelineConfig,
    device: torch.device,
) -> Any:
    if not CTGAN_AVAILABLE:
        raise RuntimeError("CTGAN package is not installed")
    if fraud_frame.empty:
        raise RuntimeError("CTGAN baseline requires at least one fraud sample")

    effective_batch_size = min(config.baselines.ctgan_batch_size, max(2, len(fraud_frame)))
    if effective_batch_size % 2 != 0 and effective_batch_size > 2:
        effective_batch_size -= 1

    model = CTGAN(
        epochs=config.baselines.ctgan_epochs,
        batch_size=effective_batch_size,
        generator_dim=tuple(config.baselines.ctgan_generator_dim),
        discriminator_dim=tuple(config.baselines.ctgan_discriminator_dim),
        verbose=False,
        enable_gpu=device.type == "cuda",
        pac=1,
    )
    model.fit(fraud_frame[feature_columns])
    return model


def _sample_ctgan(
    model: Any,
    feature_columns: list[str],
    label_column: str,
    n_samples: int,
) -> pd.DataFrame:
    if n_samples <= 0:
        return pd.DataFrame(columns=feature_columns + [label_column])
    frame = model.sample(n_samples)[feature_columns].copy()
    frame[label_column] = 1
    return frame


def _save_ddpm_model(model: DDPM, path: Path) -> str:
    ensure_directory(path.parent)
    torch.save(model.state_dict(), path)
    return str(path)


def _save_ctgan_model(model: Any, path: Path) -> str:
    ensure_directory(path.parent)
    joblib.dump(model, path)
    return str(path)


def generate_stage1_baseline_pools(
    config: PaperPipelineConfig,
    prepared_metadata: dict[str, Any],
    run_dir: str | Path,
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    run_dir = Path(run_dir)
    enabled_pool_sources = get_enabled_pool_sources(config)
    if not enabled_pool_sources:
        return {}, {}

    device = select_torch_device(config.collafuse.device)
    set_random_seed(config.sampling.random_seed)
    synthetic_root = ensure_directory(run_dir / "synthetic")
    model_root = ensure_directory(run_dir / "baseline_models")
    label_column = prepared_metadata["label_column"]
    feature_columns = prepared_metadata["feature_columns"]
    client_frames = {
        entry["client_id"]: pd.read_csv(entry["train_path"])
        for entry in prepared_metadata["clients"]
    }
    pooled_train = pd.concat(client_frames.values(), ignore_index=True)
    pooled_fraud = pooled_train.loc[pooled_train[label_column] == 1].reset_index(drop=True)

    synthetic_paths: dict[str, dict[str, str]] = {}
    model_paths: dict[str, Any] = {}

    if "local_only_ddpm" in enabled_pool_sources:
        local_paths: dict[str, str] = {}
        local_models: dict[str, str] = {}
        for client_id, train_frame in client_frames.items():
            LOGGER.info("Training local-only DDPM for %s", client_id)
            model = _train_guided_ddpm(
                train_frame,
                feature_columns,
                label_column,
                config,
                device,
                progress_desc=f"Local DDPM {client_id}",
            )
            synthetic = _sample_ddpm_model(
                model,
                feature_columns,
                label_column,
                config.sampling.samples_per_client,
                config,
                device,
                progress_desc=f"Local DDPM samples {client_id}",
            )
            path = synthetic_root / "local_only_ddpm" / f"{client_id}_synthetic.csv"
            ensure_directory(path.parent)
            synthetic.to_csv(path, index=False)
            local_paths[client_id] = str(path)
            local_models[client_id] = _save_ddpm_model(model, model_root / "local_only_ddpm" / f"{client_id}.pt")
        synthetic_paths["local_only_ddpm"] = local_paths
        model_paths["local_only_ddpm"] = local_models

    if "centralized_ddpm" in enabled_pool_sources:
        LOGGER.info("Training centralized DDPM baseline")
        model = _train_guided_ddpm(
            pooled_train,
            feature_columns,
            label_column,
            config,
            device,
            progress_desc="Centralized DDPM",
        )
        centralized_paths: dict[str, str] = {}
        for client_id in client_frames:
            synthetic = _sample_ddpm_model(
                model,
                feature_columns,
                label_column,
                config.sampling.samples_per_client,
                config,
                device,
                progress_desc=f"Centralized DDPM samples {client_id}",
            )
            path = synthetic_root / "centralized_ddpm" / f"{client_id}_synthetic.csv"
            ensure_directory(path.parent)
            synthetic.to_csv(path, index=False)
            centralized_paths[client_id] = str(path)
        synthetic_paths["centralized_ddpm"] = centralized_paths
        model_paths["centralized_ddpm"] = _save_ddpm_model(model, model_root / "centralized_ddpm.pt")

    if "tabddpm" in enabled_pool_sources:
        LOGGER.info("Training TabDDPM-style baseline")
        model = _train_standard_ddpm(
            pooled_fraud,
            feature_columns,
            config,
            device,
            progress_desc="TabDDPM",
        )
        tabddpm_paths: dict[str, str] = {}
        for client_id in client_frames:
            synthetic = _sample_ddpm_model(
                model,
                feature_columns,
                label_column,
                config.sampling.samples_per_client,
                config,
                device,
                progress_desc=f"TabDDPM samples {client_id}",
            )
            path = synthetic_root / "tabddpm" / f"{client_id}_synthetic.csv"
            ensure_directory(path.parent)
            synthetic.to_csv(path, index=False)
            tabddpm_paths[client_id] = str(path)
        synthetic_paths["tabddpm"] = tabddpm_paths
        model_paths["tabddpm"] = _save_ddpm_model(model, model_root / "tabddpm.pt")

    if "ctgan" in enabled_pool_sources:
        LOGGER.info("Training CTGAN baseline")
        try:
            ctgan_model = _fit_ctgan(pooled_fraud, feature_columns, config, device)
        except Exception as exc:  # pragma: no cover - depends on external package/runtime behavior
            LOGGER.warning("CTGAN baseline failed: %s", exc)
        else:
            ctgan_paths: dict[str, str] = {}
            for client_id in client_frames:
                synthetic = _sample_ctgan(
                    ctgan_model,
                    feature_columns,
                    label_column,
                    config.sampling.samples_per_client,
                )
                path = synthetic_root / "ctgan" / f"{client_id}_synthetic.csv"
                ensure_directory(path.parent)
                synthetic.to_csv(path, index=False)
                ctgan_paths[client_id] = str(path)
            synthetic_paths["ctgan"] = ctgan_paths
            model_paths["ctgan"] = _save_ctgan_model(ctgan_model, model_root / "ctgan.pkl")

    return synthetic_paths, model_paths
