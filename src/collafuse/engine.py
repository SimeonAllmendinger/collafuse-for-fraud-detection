from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.DDPM.DDPM import DDPM
from src.DDPM.learning_rate_schedulers import get_custom_cosine_schedule_with_warmup
from src.DDPM.losses import fraud_diffuse_loss
from src.DDPM.time_embedding import get_sinusoidal_embedding
from src.collafuse.dataset import TabularFraudDataset
from src.config_files.configs import PaperPipelineConfig
from src.pipeline.common import ensure_directory, select_torch_device, set_random_seed

LOGGER = logging.getLogger("collafuse")


@dataclass
class ClientRuntime:
    client_id: str
    client_label: str
    train_frame: pd.DataFrame
    test_frame: pd.DataFrame
    dataset: TabularFraudDataset
    dataloader: DataLoader
    model: DDPM
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    fraud_pool: torch.Tensor
    nonfraud_pool: torch.Tensor
    nonfraud_mean: torch.Tensor
    nonfraud_std: torch.Tensor


def _noise_sign_accuracy(predicted_noise: torch.Tensor, true_noise: torch.Tensor) -> float:
    predicted_sign = predicted_noise >= 0
    true_sign = true_noise >= 0
    return float((predicted_sign == true_sign).float().mean().item())


class CollaFuseExperiment:
    def __init__(self, config: PaperPipelineConfig, prepared_metadata: dict[str, Any]):
        self.config = config
        self.prepared_metadata = prepared_metadata
        self.device = select_torch_device(config.collafuse.device)
        self.label_column = prepared_metadata["label_column"]
        self.feature_columns = prepared_metadata["feature_columns"]
        self.cut_timestep = int(round(config.collafuse.tau_ratio * config.collafuse.num_timesteps))
        self.clients = self._build_clients(prepared_metadata["clients"])
        self.cloud_model = DDPM(
            device=self.device,
            input_dim=len(self.feature_columns),
            T=config.collafuse.num_timesteps,
            beta_start=config.collafuse.beta_start,
            beta_end=config.collafuse.beta_end,
            time_embed_dim=config.collafuse.time_embed_dim,
        ).to(self.device)
        total_steps = max(1, sum(len(client.dataloader) for client in self.clients.values()) * config.collafuse.epochs)
        self.cloud_optimizer = torch.optim.AdamW(self.cloud_model.parameters(), lr=config.collafuse.learning_rate)
        self.cloud_scheduler = get_custom_cosine_schedule_with_warmup(
            optimizer=self.cloud_optimizer,
            num_warmup_steps=config.collafuse.lr_warmup_steps,
            num_training_steps=total_steps,
        )

    def _conditioning_embedding(self, model: DDPM, labels: torch.Tensor) -> torch.Tensor:
        return model.class_embedding(labels.long())

    def _target_labels(self, batch_size: int) -> torch.Tensor:
        return torch.full(
            (batch_size,),
            int(self.config.sampling.target_label),
            device=self.device,
            dtype=torch.long,
        )

    def _build_clients(self, client_entries: list[dict[str, Any]]) -> dict[str, ClientRuntime]:
        clients: dict[str, ClientRuntime] = {}
        total_steps = 0
        for entry in client_entries:
            train_frame = pd.read_csv(entry["train_path"])
            test_frame = pd.read_csv(entry["test_path"])
            dataset = TabularFraudDataset(train_frame, label_column=self.label_column)
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.collafuse.batch_size,
                shuffle=True,
                num_workers=self.config.collafuse.num_workers,
            )
            model = DDPM(
                device=self.device,
                input_dim=len(dataset.feature_columns),
                T=self.config.collafuse.num_timesteps,
                beta_start=self.config.collafuse.beta_start,
                beta_end=self.config.collafuse.beta_end,
                time_embed_dim=self.config.collafuse.time_embed_dim,
            ).to(self.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.collafuse.learning_rate)
            total_steps += len(dataloader) * self.config.collafuse.epochs
            fraud_pool = torch.tensor(
                train_frame.loc[train_frame[self.label_column] == 1, dataset.feature_columns].to_numpy(dtype="float32"),
                device=self.device,
            )
            nonfraud_pool = torch.tensor(
                train_frame.loc[train_frame[self.label_column] == 0, dataset.feature_columns].to_numpy(dtype="float32"),
                device=self.device,
            )
            nonfraud_mean = (
                nonfraud_pool.mean(dim=0, keepdim=True)
                if len(nonfraud_pool) > 0
                else torch.zeros((1, len(dataset.feature_columns)), device=self.device)
            )
            nonfraud_std = (
                nonfraud_pool.std(dim=0, keepdim=True, unbiased=False) + 1e-5
                if len(nonfraud_pool) > 0
                else torch.ones((1, len(dataset.feature_columns)), device=self.device)
            )
            clients[entry["client_id"]] = ClientRuntime(
                client_id=entry["client_id"],
                client_label=entry["client_label"],
                train_frame=train_frame,
                test_frame=test_frame,
                dataset=dataset,
                dataloader=dataloader,
                model=model,
                optimizer=optimizer,
                scheduler=None,  # type: ignore[assignment]
                fraud_pool=fraud_pool,
                nonfraud_pool=nonfraud_pool,
                nonfraud_mean=nonfraud_mean,
                nonfraud_std=nonfraud_std,
            )
        for client in clients.values():
            client.scheduler = get_custom_cosine_schedule_with_warmup(
                optimizer=client.optimizer,
                num_warmup_steps=self.config.collafuse.lr_warmup_steps,
                num_training_steps=max(1, len(client.dataloader) * self.config.collafuse.epochs),
            )
        return clients

    def _sample_triplets(
        self,
        client: ClientRuntime,
        batch_size: int
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        triplet_batch_size = min(batch_size, len(client.fraud_pool), len(client.nonfraud_pool))
        if triplet_batch_size == 0:
            return None, None, None

        fraud_indices = torch.randint(0, len(client.fraud_pool), (triplet_batch_size,), device=self.device)
        nonfraud_indices = torch.randint(0, len(client.nonfraud_pool), (triplet_batch_size,), device=self.device)
        positive = client.fraud_pool[fraud_indices]
        negative = client.nonfraud_pool[nonfraud_indices]

        z = torch.randn((triplet_batch_size, len(self.feature_columns)), device=self.device)
        x_t = client.nonfraud_mean + client.nonfraud_std * z
        zero_steps = torch.zeros((triplet_batch_size,), dtype=torch.long, device=self.device)
        t_emb = get_sinusoidal_embedding(zero_steps, embedding_dim=self.config.collafuse.time_embed_dim).to(self.device)
        target_cond = self._conditioning_embedding(client.model, self._target_labels(triplet_batch_size))
        noise_prediction = client.model.noise_predictor(x_t, t_emb, target_cond)
        sqrt_alpha_bar_0 = client.model.beta_schedule.sqrt_alpha_bars[0]
        one_minus_alpha_bar_0 = client.model.beta_schedule.one_minus_alpha_bars[0]
        anchor = (x_t - one_minus_alpha_bar_0 * noise_prediction) / sqrt_alpha_bar_0
        return anchor, positive, negative

    def _train_client_epoch(self, client: ClientRuntime, epoch: int) -> dict[str, float]:
        client.model.train()
        self.cloud_model.train()
        client_loss_total = 0.0
        cloud_loss_total = 0.0
        client_accuracy_total = 0.0
        cloud_accuracy_total = 0.0
        l_norm_total = 0.0
        l_prior_total = 0.0
        l_triplet_total = 0.0
        client_steps = 0
        cloud_steps = 0

        with tqdm(
            total=len(client.dataloader),
            desc=f"Epoch {epoch}/{self.config.collafuse.epochs} {client.client_id}",
            leave=False,
        ) as batch_progress:
            for batch_features, batch_labels in client.dataloader:
                x_0 = batch_features.to(self.device)
                labels = batch_labels.to(self.device, dtype=torch.long)
                batch_size = x_0.shape[0]
                client_noise = torch.randn_like(x_0)
                anchor, positive, negative = self._sample_triplets(client, batch_size)

                if self.cut_timestep > 0:
                    client_timesteps = torch.randint(
                        0, max(1, self.cut_timestep), (batch_size,), device=self.device, dtype=torch.long
                    )
                    sqrt_alpha = client.model.beta_schedule.sqrt_alpha_bars[client_timesteps].unsqueeze(1)
                    sqrt_one_minus_alpha = client.model.beta_schedule.one_minus_alpha_bars[
                        client_timesteps].unsqueeze(1)
                    x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * client_noise
                    t_emb = get_sinusoidal_embedding(
                        client_timesteps,
                        embedding_dim=self.config.collafuse.time_embed_dim
                    ).to(self.device)
                    client_cond = self._conditioning_embedding(client.model, labels)
                    predicted_noise = client.model.noise_predictor(x_t, t_emb, client_cond)
                    loss, components = fraud_diffuse_loss(
                        predicted_noise=predicted_noise,
                        true_noise=client_noise,
                        mu_nf=client.nonfraud_mean,
                        sigma_nf=client.nonfraud_std,
                        anchor=anchor,
                        positive=positive,
                        negative=negative,
                        w1=self.config.collafuse.w1,
                        w2=self.config.collafuse.w2,
                    )
                    client.optimizer.zero_grad()
                    loss.backward()
                    client.optimizer.step()
                    client.scheduler.step()
                    client_loss_total += float(loss.detach().item())
                    client_accuracy_total += _noise_sign_accuracy(predicted_noise.detach(), client_noise)
                    l_norm_total += components["L_norm"]
                    l_prior_total += components["L_prior"]
                    l_triplet_total += components["L_triplet"]
                    client_steps += 1

                if self.cut_timestep < self.config.collafuse.num_timesteps:
                    cloud_noise = torch.randn_like(x_0)
                    if self.cut_timestep > 0:
                        cut_index = min(self.cut_timestep, self.config.collafuse.num_timesteps - 1)
                        cut_steps = torch.full((batch_size,), cut_index, device=self.device, dtype=torch.long)
                        sqrt_alpha_cut = client.model.beta_schedule.sqrt_alpha_bars[cut_steps].unsqueeze(1)
                        sqrt_one_minus_cut = client.model.beta_schedule.one_minus_alpha_bars[cut_steps].unsqueeze(1)
                        x_cut = sqrt_alpha_cut * x_0 + sqrt_one_minus_cut * client_noise
                    else:
                        x_cut = x_0

                    cloud_timesteps = torch.randint(
                        min(self.cut_timestep, self.config.collafuse.num_timesteps - 1),
                        self.config.collafuse.num_timesteps,
                        (batch_size,),
                        device=self.device,
                        dtype=torch.long,
                    )
                    sqrt_alpha_cloud = self.cloud_model.beta_schedule.sqrt_alpha_bars[cloud_timesteps].unsqueeze(1)
                    sqrt_one_minus_cloud = self.cloud_model.beta_schedule.one_minus_alpha_bars[
                        cloud_timesteps].unsqueeze(1)
                    x_cloud = sqrt_alpha_cloud * x_cut + sqrt_one_minus_cloud * cloud_noise
                    t_emb = get_sinusoidal_embedding(
                        cloud_timesteps,
                        embedding_dim=self.config.collafuse.time_embed_dim
                    ).to(self.device)
                    cloud_cond = self._conditioning_embedding(self.cloud_model, labels)
                    cloud_prediction = self.cloud_model.noise_predictor(x_cloud, t_emb, cloud_cond)
                    cloud_loss = F.mse_loss(cloud_prediction, cloud_noise)
                    self.cloud_optimizer.zero_grad()
                    cloud_loss.backward()
                    self.cloud_optimizer.step()
                    self.cloud_scheduler.step()
                    cloud_loss_total += float(cloud_loss.detach().item())
                    cloud_accuracy_total += _noise_sign_accuracy(cloud_prediction.detach(), cloud_noise)
                    cloud_steps += 1

                batch_progress.update(1)
                batch_progress.set_postfix(
                    client_loss=f"{client_loss_total / max(1, client_steps):.4f}",
                    cloud_loss=f"{cloud_loss_total / max(1, cloud_steps):.4f}",
                )

        return {
            "epoch": epoch,
            "client_id": client.client_id,
            "client_loss": client_loss_total / max(1, client_steps),
            "cloud_loss": cloud_loss_total / max(1, cloud_steps),
            "client_noise_accuracy": client_accuracy_total / max(1, client_steps),
            "cloud_noise_accuracy": cloud_accuracy_total / max(1, cloud_steps),
            "l_norm": l_norm_total / max(1, client_steps),
            "l_prior": l_prior_total / max(1, client_steps),
            "l_triplet": l_triplet_total / max(1, client_steps),
        }

    def save_checkpoint(self, checkpoint_path: str | Path, epoch: int) -> Path:
        checkpoint_path = Path(checkpoint_path)
        ensure_directory(checkpoint_path.parent)
        torch.save(
            {
                "epoch": epoch,
                "cloud_model": self.cloud_model.state_dict(),
                "client_models": {client_id: client.model.state_dict() for client_id, client in self.clients.items()},
                "feature_columns": self.feature_columns,
                "cut_timestep": self.cut_timestep,
            },
            checkpoint_path,
        )
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str | Path) -> int:
        payload = torch.load(checkpoint_path, map_location=self.device)
        self.cloud_model.load_state_dict(payload["cloud_model"])
        for client_id, client_state in payload["client_models"].items():
            self.clients[client_id].model.load_state_dict(client_state)
        return int(payload.get("epoch", 0))

    def train(self, run_dir: str | Path, reuse_checkpoint: bool = False) -> tuple[Path, pd.DataFrame]:
        run_dir = Path(run_dir)
        checkpoint_path = run_dir / "checkpoints" / "collafuse.pt"
        if reuse_checkpoint and checkpoint_path.exists():
            LOGGER.info("Reusing existing Stage 1 checkpoint at %s", checkpoint_path)
            self.load_checkpoint(checkpoint_path)
            history = pd.read_csv(
                run_dir / "training_history.csv"
            ) if (run_dir / "training_history.csv").exists() else pd.DataFrame()
            return checkpoint_path, history

        set_random_seed(self.config.sampling.random_seed)
        LOGGER.info(
            "Starting CollaFuse training with %s client(s), %s epoch(s), device=%s",
            len(self.clients),
            self.config.collafuse.epochs,
            self.device,
        )
        history_rows: list[dict[str, float]] = []
        with tqdm(range(1, self.config.collafuse.epochs + 1), desc="Stage 1 training", leave=True) as epoch_progress:
            for epoch in epoch_progress:
                epoch_results: list[dict[str, float]] = []
                for client in self.clients.values():
                    epoch_results.append(self._train_client_epoch(client, epoch))
                history_rows.extend(epoch_results)
                avg_client_loss = sum(result["client_loss"] for result in epoch_results) / max(1, len(epoch_results))
                avg_cloud_loss = sum(result["cloud_loss"] for result in epoch_results) / max(1, len(epoch_results))
                epoch_progress.set_postfix(client_loss=f"{avg_client_loss:.4f}", cloud_loss=f"{avg_cloud_loss:.4f}")
                LOGGER.info(
                    "Finished epoch %s/%s | avg_client_loss=%.4f | avg_cloud_loss=%.4f",
                    epoch,
                    self.config.collafuse.epochs,
                    avg_client_loss,
                    avg_cloud_loss,
                )
                if epoch % self.config.collafuse.checkpoint_every == 0 or epoch == self.config.collafuse.epochs:
                    self.save_checkpoint(checkpoint_path, epoch)
                    LOGGER.info("Saved checkpoint for epoch %s to %s", epoch, checkpoint_path)

        history = pd.DataFrame(history_rows)
        history.to_csv(run_dir / "training_history.csv", index=False)
        LOGGER.info("Training history saved to %s", run_dir / "training_history.csv")
        return checkpoint_path, history

    @torch.no_grad()
    def sample_client(self, client: ClientRuntime, n_samples: int, progress_desc: str | None = None) -> pd.DataFrame:
        client.model.eval()
        self.cloud_model.eval()
        sample_batch_size = self.config.sampling.sample_batch_size
        total_steps = self.config.sampling.n_inference_timesteps or self.config.collafuse.num_timesteps
        outputs = []
        batch_starts = list(range(0, n_samples, sample_batch_size))
        total_progress_steps = len(batch_starts) * total_steps
        with tqdm(
            total=total_progress_steps,
            desc=progress_desc or f"CollaFuse samples {client.client_id}",
            leave=True,
            disable=progress_desc is None,
        ) as progress:
            for start in batch_starts:
                current_batch_size = min(sample_batch_size, n_samples - start)
                x_t = torch.randn((current_batch_size, len(self.feature_columns)), device=self.device)
                target_labels = self._target_labels(current_batch_size)
                for t_idx in reversed(range(total_steps)):
                    timesteps = torch.full((current_batch_size,), t_idx, device=self.device, dtype=torch.long)
                    t_emb = get_sinusoidal_embedding(
                        timesteps,
                        embedding_dim=self.config.collafuse.time_embed_dim
                    ).to(self.device)
                    if t_idx >= self.cut_timestep:
                        cond = self._conditioning_embedding(self.cloud_model, target_labels)
                        x_t = self.cloud_model.p_sample(x_t, timesteps, t_emb, cond=cond)
                    else:
                        cond = self._conditioning_embedding(client.model, target_labels)
                        x_t = client.model.p_sample(x_t, timesteps, t_emb, cond=cond)
                    progress.update(1)
                outputs.append(x_t.detach().cpu())
        frame = pd.DataFrame(torch.cat(outputs, dim=0).numpy(), columns=self.feature_columns)
        frame[self.label_column] = 1
        return frame

    def sample_all_clients(self, run_dir: str | Path) -> dict[str, str]:
        run_dir = Path(run_dir)
        synthetic_dir = ensure_directory(run_dir / "synthetic")
        paths: dict[str, str] = {}
        for client_id, client in self.clients.items():
            LOGGER.info("Generating CollaFuse synthetic samples for %s", client_id)
            synthetic = self.sample_client(
                client,
                self.config.sampling.samples_per_client,
                progress_desc=f"CollaFuse sampling {client_id}",
            )
            path = synthetic_dir / f"{client_id}_synthetic.csv"
            synthetic.to_csv(path, index=False)
            paths[client_id] = str(path)
            LOGGER.info("Saved CollaFuse synthetic samples for %s to %s", client_id, path)
        return paths
