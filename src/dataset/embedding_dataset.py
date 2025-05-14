"""This module contains the EmbeddingDataset class."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class NoisyEmbeddingDataset(Dataset):
    """Loads the data from the given path and returns a subset of the data."""

    def __init__(
        self,
        data: np.ndarray,
    ) -> None:
        """Initializes the dataset.

        Args:
            data: The noisy embeddings.
        """
        self.data = data

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns the item at the given index."""
        emb = self.data[idx]
        if not isinstance(emb, torch.Tensor):
            emb = torch.tensor(emb, dtype=torch.float)
        return emb
