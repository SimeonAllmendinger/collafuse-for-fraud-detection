from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.collafuse.visualization import (
    plot_collafuse_training_accuracy,
    plot_collafuse_training_loss,
)


class VisualizationTest(unittest.TestCase):
    def test_stage1_training_plots_are_written(self) -> None:
        history = pd.DataFrame(
            [
                {
                    "epoch": 1,
                    "client_id": "CLIENT_1",
                    "client_loss": 1.2,
                    "cloud_loss": 0.9,
                    "client_noise_accuracy": 0.61,
                    "cloud_noise_accuracy": 0.58,
                    "l_norm": 0.4,
                    "l_prior": 0.3,
                    "l_triplet": 0.5,
                },
                {
                    "epoch": 1,
                    "client_id": "CLIENT_2",
                    "client_loss": 1.1,
                    "cloud_loss": 0.88,
                    "client_noise_accuracy": 0.63,
                    "cloud_noise_accuracy": 0.59,
                    "l_norm": 0.38,
                    "l_prior": 0.28,
                    "l_triplet": 0.44,
                },
                {
                    "epoch": 2,
                    "client_id": "CLIENT_1",
                    "client_loss": 0.95,
                    "cloud_loss": 0.76,
                    "client_noise_accuracy": 0.69,
                    "cloud_noise_accuracy": 0.64,
                    "l_norm": 0.32,
                    "l_prior": 0.24,
                    "l_triplet": 0.39,
                },
                {
                    "epoch": 2,
                    "client_id": "CLIENT_2",
                    "client_loss": 0.92,
                    "cloud_loss": 0.74,
                    "client_noise_accuracy": 0.71,
                    "cloud_noise_accuracy": 0.66,
                    "l_norm": 0.31,
                    "l_prior": 0.23,
                    "l_triplet": 0.36,
                },
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            loss_path = tmp_path / "collafuse_training_loss.png"
            accuracy_path = tmp_path / "collafuse_training_accuracy.png"

            plot_collafuse_training_loss(history, loss_path)
            plot_collafuse_training_accuracy(history, accuracy_path)

            self.assertTrue(loss_path.exists())
            self.assertTrue(accuracy_path.exists())


if __name__ == "__main__":
    unittest.main()
