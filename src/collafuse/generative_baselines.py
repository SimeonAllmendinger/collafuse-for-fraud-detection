from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from src.DDPM.beta_schedule import BetaSchedule
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


@dataclass(frozen=True)
class TabDDPMFeatureLayout:
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    categorical_groups: list[tuple[str, list[str]]]

    @property
    def categorical_group_sizes(self) -> list[int]:
        return [len(group_columns) for _group_name, group_columns in self.categorical_groups]

    @property
    def categorical_feature_columns(self) -> list[str]:
        return [column for _group_name, group_columns in self.categorical_groups for column in group_columns]


class TabDDPMDenoiser(nn.Module):
    def __init__(
        self,
        input_dim: int,
        time_embed_dim: int,
        num_classes: int = 2,
        class_embed_dim: int = 64,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.class_embedding = nn.Embedding(num_classes, class_embed_dim)
        self.network = nn.Sequential(
            nn.Linear(input_dim + hidden_dim + class_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        t_emb = get_sinusoidal_embedding(timesteps, embedding_dim=self.time_embed_dim).to(x.device)
        t_proj = self.time_mlp(t_emb)
        y_proj = self.class_embedding(labels.long())
        return self.network(torch.cat([x, t_proj, y_proj], dim=1))


def _load_preprocessing_template(prepared_metadata: dict[str, Any]) -> dict[str, Any]:
    template_path = prepared_metadata.get("template_path")
    if not template_path:
        raise ValueError("Prepared metadata is missing template_path required for the TabDDPM baseline")
    with Path(template_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_tabddpm_feature_layout(
    feature_columns: list[str],
    prepared_metadata: dict[str, Any],
) -> TabDDPMFeatureLayout:
    template = _load_preprocessing_template(prepared_metadata)
    feature_set = set(feature_columns)
    numeric_columns = [column for column in template["numeric_columns"] if column in feature_set]
    encoded_columns = set(template["encoded_columns"])

    categorical_groups: list[tuple[str, list[str]]] = []
    for categorical_column in template["categorical_columns"]:
        prefix = f"{categorical_column}_"
        group_columns = [
            column
            for column in feature_columns
            if column in encoded_columns and column.startswith(prefix)
        ]
        if group_columns:
            categorical_groups.append((categorical_column, group_columns))

    categorical_feature_columns = {
        column
        for _group_name, group_columns in categorical_groups
        for column in group_columns
    }
    ordered_feature_columns = numeric_columns + [
        column for column in feature_columns if column in categorical_feature_columns
    ]
    return TabDDPMFeatureLayout(
        feature_columns=ordered_feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=template["categorical_columns"],
        categorical_groups=categorical_groups,
    )


def _fit_tabddpm_numeric_transformer(
    frame: pd.DataFrame,
    numeric_columns: list[str],
    random_state: int,
) -> QuantileTransformer | None:
    if not numeric_columns:
        return None
    n_quantiles = max(2, min(1000, len(frame)))
    transformer = QuantileTransformer(
        n_quantiles=n_quantiles,
        output_distribution="normal",
        random_state=random_state,
    )
    transformer.fit(frame[numeric_columns])
    return transformer


def _tensor_from_frame(frame: pd.DataFrame, columns: list[str], device: torch.device) -> torch.Tensor:
    if not columns:
        return torch.zeros((len(frame), 0), dtype=torch.float32, device=device)
    return torch.tensor(frame[columns].to_numpy(dtype="float32"), dtype=torch.float32, device=device)


def _apply_tabddpm_numeric_transform(
    frame: pd.DataFrame,
    numeric_columns: list[str],
    transformer: QuantileTransformer | None,
    device: torch.device,
) -> torch.Tensor:
    if not numeric_columns:
        return torch.zeros((len(frame), 0), dtype=torch.float32, device=device)
    numeric_frame = frame[numeric_columns].to_numpy(dtype="float32")
    if transformer is not None:
        numeric_frame = transformer.transform(numeric_frame).astype("float32")
    return torch.tensor(numeric_frame, dtype=torch.float32, device=device)


def _q_sample_tabddpm_categorical(
    x_start_cat: torch.Tensor,
    timesteps: torch.Tensor,
    beta_schedule: BetaSchedule,
    categorical_group_sizes: list[int],
) -> torch.Tensor:
    if x_start_cat.shape[1] == 0:
        return x_start_cat

    batch_size = x_start_cat.shape[0]
    keep_prob = beta_schedule.alpha_bars[timesteps]
    x_cat_t = torch.zeros_like(x_start_cat)
    start = 0
    for group_size in categorical_group_sizes:
        stop = start + group_size
        group = x_start_cat[:, start:stop]
        original_index = group.argmax(dim=1)
        random_index = torch.randint(0, group_size, (batch_size,), device=x_start_cat.device)
        keep_mask = torch.rand(batch_size, device=x_start_cat.device) < keep_prob
        sampled_index = torch.where(keep_mask, original_index, random_index)
        x_cat_t[:, start:stop] = F.one_hot(sampled_index, num_classes=group_size).float()
        start = stop
    return x_cat_t


def _tabddpm_categorical_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    categorical_group_sizes: list[int],
) -> torch.Tensor:
    if logits.shape[1] == 0 or not categorical_group_sizes:
        return torch.tensor(0.0, device=logits.device)

    losses = []
    start = 0
    for group_size in categorical_group_sizes:
        stop = start + group_size
        group_targets = targets[:, start:stop].argmax(dim=1)
        losses.append(F.cross_entropy(logits[:, start:stop], group_targets))
        start = stop
    return torch.stack(losses).mean()


def _sample_tabddpm_categorical(
    logits: torch.Tensor,
    categorical_group_sizes: list[int],
) -> torch.Tensor:
    if logits.shape[1] == 0 or not categorical_group_sizes:
        return logits.new_zeros((logits.shape[0], 0))

    samples: list[torch.Tensor] = []
    start = 0
    for group_size in categorical_group_sizes:
        stop = start + group_size
        probs = torch.softmax(logits[:, start:stop], dim=1)
        sampled_index = torch.multinomial(probs, num_samples=1).squeeze(1)
        samples.append(F.one_hot(sampled_index, num_classes=group_size).float())
        start = stop
    return torch.cat(samples, dim=1)


def _tabddpm_gaussian_step(
    x_t: torch.Tensor,
    predicted_noise: torch.Tensor,
    timesteps: torch.Tensor,
    beta_schedule: BetaSchedule,
) -> torch.Tensor:
    if x_t.shape[1] == 0:
        return x_t

    alpha_t = beta_schedule.alphas[timesteps].unsqueeze(1)
    beta_t = beta_schedule.betas[timesteps].unsqueeze(1)
    alpha_bar_t = beta_schedule.alpha_bars[timesteps].unsqueeze(1)
    sqrt_one_minus_alpha_bar_t = beta_schedule.one_minus_alpha_bars[timesteps].unsqueeze(1)
    mean = (x_t - (beta_t / sqrt_one_minus_alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_t)
    variance = beta_schedule.posterior_variance[timesteps].unsqueeze(1)
    noise = torch.randn_like(x_t)
    nonzero_mask = (timesteps > 0).float().unsqueeze(1)
    return mean + nonzero_mask * torch.sqrt(variance) * noise


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


def _train_tabddpm(
    frame: pd.DataFrame,
    label_column: str,
    layout: TabDDPMFeatureLayout,
    config: PaperPipelineConfig,
    device: torch.device,
    progress_desc: str,
) -> tuple[TabDDPMDenoiser, QuantileTransformer | None]:
    numeric_transformer = _fit_tabddpm_numeric_transformer(
        frame,
        layout.numeric_columns,
        random_state=config.sampling.random_seed,
    )
    numeric_features = _apply_tabddpm_numeric_transform(
        frame,
        layout.numeric_columns,
        numeric_transformer,
        device=device,
    ).cpu()
    categorical_features = _tensor_from_frame(frame, layout.categorical_feature_columns, device=device).cpu()
    labels = torch.tensor(frame[label_column].to_numpy(dtype="int64"), dtype=torch.long)
    dataloader = DataLoader(
        TensorDataset(numeric_features, categorical_features, labels),
        batch_size=_ddpm_batch_size(config),
        shuffle=True,
        num_workers=config.collafuse.num_workers,
    )

    input_dim = len(layout.numeric_columns) + len(layout.categorical_feature_columns)
    hidden_dim = max(128, min(512, max(1, input_dim) * 4))
    model = TabDDPMDenoiser(
        input_dim=input_dim,
        time_embed_dim=config.collafuse.time_embed_dim,
        hidden_dim=hidden_dim,
    ).to(device)
    beta_schedule = BetaSchedule(
        device=device,
        T=config.collafuse.num_timesteps,
        beta_start=config.collafuse.beta_start,
        beta_end=config.collafuse.beta_end,
    )
    categorical_group_sizes = layout.categorical_group_sizes
    num_numeric = len(layout.numeric_columns)

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
            for batch_numeric, batch_categorical, batch_labels in dataloader:
                x_num = batch_numeric.to(device)
                x_cat = batch_categorical.to(device)
                labels_batch = batch_labels.to(device)
                batch_size = labels_batch.shape[0]
                timesteps = torch.randint(
                    0,
                    config.collafuse.num_timesteps,
                    (batch_size,),
                    device=device,
                    dtype=torch.long,
                )

                num_noise = torch.randn_like(x_num)
                if num_numeric:
                    sqrt_alpha = beta_schedule.sqrt_alpha_bars[timesteps].unsqueeze(1)
                    sqrt_one_minus_alpha = beta_schedule.one_minus_alpha_bars[timesteps].unsqueeze(1)
                    x_num_t = sqrt_alpha * x_num + sqrt_one_minus_alpha * num_noise
                else:
                    x_num_t = x_num
                x_cat_t = _q_sample_tabddpm_categorical(x_cat, timesteps, beta_schedule, categorical_group_sizes)
                model_input = torch.cat([x_num_t, x_cat_t], dim=1)
                model_output = model(model_input, timesteps, labels_batch)
                predicted_noise = model_output[:, :num_numeric]
                predicted_cat_logits = model_output[:, num_numeric:]

                numeric_loss = (
                    F.mse_loss(predicted_noise, num_noise)
                    if num_numeric
                    else torch.tensor(0.0, device=device)
                )
                categorical_loss = _tabddpm_categorical_loss(
                    predicted_cat_logits,
                    x_cat,
                    categorical_group_sizes,
                )
                loss = numeric_loss + categorical_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                loss_total += float(loss.detach().item())
                steps += 1
            epoch_progress.set_postfix(loss=f"{loss_total / max(1, steps):.4f}")
    return model.eval(), numeric_transformer


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


@torch.no_grad()
def _sample_tabddpm_model(
    model: TabDDPMDenoiser,
    numeric_transformer: QuantileTransformer | None,
    layout: TabDDPMFeatureLayout,
    label_column: str,
    target_label: int,
    n_samples: int,
    config: PaperPipelineConfig,
    device: torch.device,
    progress_desc: str,
) -> pd.DataFrame:
    if n_samples <= 0:
        return pd.DataFrame(columns=layout.feature_columns + [label_column])

    beta_schedule = BetaSchedule(
        device=device,
        T=config.collafuse.num_timesteps,
        beta_start=config.collafuse.beta_start,
        beta_end=config.collafuse.beta_end,
    )
    batch_size = config.sampling.sample_batch_size
    total_batches = list(range(0, n_samples, batch_size))
    outputs: list[pd.DataFrame] = []
    num_numeric = len(layout.numeric_columns)
    categorical_group_sizes = layout.categorical_group_sizes

    with tqdm(total=len(total_batches), desc=progress_desc, leave=True) as progress:
        for start in total_batches:
            current_batch_size = min(batch_size, n_samples - start)
            labels = torch.full((current_batch_size,), int(target_label), device=device, dtype=torch.long)
            x_num = torch.randn((current_batch_size, num_numeric), device=device) if num_numeric else torch.zeros((current_batch_size, 0), device=device)
            cat_parts = []
            for group_size in categorical_group_sizes:
                sampled_index = torch.randint(0, group_size, (current_batch_size,), device=device)
                cat_parts.append(F.one_hot(sampled_index, num_classes=group_size).float())
            x_cat = torch.cat(cat_parts, dim=1) if cat_parts else torch.zeros((current_batch_size, 0), device=device)

            num_steps = config.sampling.n_inference_timesteps or config.collafuse.num_timesteps
            for t_idx in reversed(range(num_steps)):
                timesteps = torch.full((current_batch_size,), t_idx, device=device, dtype=torch.long)
                model_input = torch.cat([x_num, x_cat], dim=1)
                model_output = model(model_input, timesteps, labels)
                predicted_noise = model_output[:, :num_numeric]
                predicted_cat_logits = model_output[:, num_numeric:]
                x_num = _tabddpm_gaussian_step(x_num, predicted_noise, timesteps, beta_schedule)
                x_cat = _sample_tabddpm_categorical(predicted_cat_logits, categorical_group_sizes)

            if num_numeric:
                sampled_numeric = x_num.detach().cpu().numpy()
                if numeric_transformer is not None:
                    sampled_numeric = numeric_transformer.inverse_transform(sampled_numeric)
                sampled_numeric = np.clip(sampled_numeric, 0.0, 1.0)
                numeric_frame = pd.DataFrame(sampled_numeric, columns=layout.numeric_columns)
            else:
                numeric_frame = pd.DataFrame(index=range(current_batch_size))

            if layout.categorical_feature_columns:
                categorical_frame = pd.DataFrame(
                    x_cat.detach().cpu().numpy(),
                    columns=layout.categorical_feature_columns,
                )
            else:
                categorical_frame = pd.DataFrame(index=range(current_batch_size))

            combined = pd.concat([numeric_frame, categorical_frame], axis=1)
            combined = combined.reindex(columns=layout.feature_columns, fill_value=0.0)
            combined[label_column] = int(target_label)
            outputs.append(combined)
            progress.update(1)

    return pd.concat(outputs, ignore_index=True)


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


def _save_tabddpm_model(
    model: TabDDPMDenoiser,
    numeric_transformer: QuantileTransformer | None,
    layout: TabDDPMFeatureLayout,
    path: Path,
) -> str:
    ensure_directory(path.parent)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "numeric_transformer": numeric_transformer,
            "layout": {
                "feature_columns": layout.feature_columns,
                "numeric_columns": layout.numeric_columns,
                "categorical_columns": layout.categorical_columns,
                "categorical_groups": layout.categorical_groups,
            },
        },
        path,
    )
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
    tabddpm_layout = _build_tabddpm_feature_layout(feature_columns, prepared_metadata)

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
        LOGGER.info("Training client-specific TabDDPM-style baselines")
        tabddpm_paths: dict[str, str] = {}
        tabddpm_model_paths: dict[str, str] = {}
        for client_id, train_frame in client_frames.items():
            LOGGER.info("Training TabDDPM-style baseline for %s", client_id)
            model, numeric_transformer = _train_tabddpm(
                train_frame,
                label_column,
                tabddpm_layout,
                config,
                device,
                progress_desc=f"TabDDPM {client_id}",
            )
            synthetic = _sample_tabddpm_model(
                model,
                numeric_transformer,
                tabddpm_layout,
                label_column,
                config.sampling.target_label,
                config.sampling.samples_per_client,
                config,
                device,
                progress_desc=f"TabDDPM samples {client_id}",
            )
            path = synthetic_root / "tabddpm" / f"{client_id}_synthetic.csv"
            ensure_directory(path.parent)
            synthetic.to_csv(path, index=False)
            tabddpm_paths[client_id] = str(path)
            tabddpm_model_paths[client_id] = _save_tabddpm_model(
                model,
                numeric_transformer,
                tabddpm_layout,
                model_root / "tabddpm" / f"{client_id}.pt",
            )
        synthetic_paths["tabddpm"] = tabddpm_paths
        model_paths["tabddpm"] = tabddpm_model_paths

    if "ctgan" in enabled_pool_sources:
        LOGGER.info("Training client-specific CTGAN baselines")
        ctgan_paths: dict[str, str] = {}
        ctgan_model_paths: dict[str, str] = {}
        for client_id, train_frame in client_frames.items():
            client_fraud = train_frame.loc[train_frame[label_column] == 1].reset_index(drop=True)
            try:
                ctgan_model = _fit_ctgan(client_fraud, feature_columns, config, device)
            except Exception as exc:  # pragma: no cover - depends on external package/runtime behavior
                LOGGER.warning("CTGAN baseline failed for %s: %s", client_id, exc)
            else:
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
                ctgan_model_paths[client_id] = _save_ctgan_model(
                    ctgan_model,
                    model_root / "ctgan" / f"{client_id}.pkl",
                )
        if ctgan_paths:
            synthetic_paths["ctgan"] = ctgan_paths
            model_paths["ctgan"] = ctgan_model_paths

    return synthetic_paths, model_paths
