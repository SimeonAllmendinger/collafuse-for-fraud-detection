from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


class TabularFraudDataset(Dataset):
    def __init__(self, data: str | Path | pd.DataFrame, label_column: str = "isFraud"):
        if isinstance(data, (str, Path)):
            frame = pd.read_csv(data)
        else:
            frame = data.copy()

        self.label_column = label_column
        self.feature_columns = [column for column in frame.columns if column != label_column]
        self.frame = frame
        self.X = frame[self.feature_columns].astype("float32").to_numpy()
        self.y = frame[label_column].astype("int64").to_numpy() if label_column in frame.columns else None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int):
        features = torch.tensor(self.X[index], dtype=torch.float32)
        if self.y is None:
            return features
        label = torch.tensor(self.y[index], dtype=torch.long)
        return features, label
