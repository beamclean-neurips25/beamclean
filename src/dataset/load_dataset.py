from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from src.dataset.base_dataset import (
    BaseDataset,
)
from src.dataset.hellaswag import HellaSwagDataset
from src.dataset.mrpc import MRPC_Dataset
from src.dataset.openorca import OpenOrcaDataset
from src.dataset.papillon import PapillonDataset

if TYPE_CHECKING:
    import omegaconf


def get_dataset(config: omegaconf.DictConfig) -> BaseDataset:
    """Create the dataset based on the configuration."""
    # If config.data has the keys batch_col, and inner_batch_col,
    # pass them to the dataset
    logging.info("Loading dataset: %s", config.data.name)
    if config.data.name == "mrpc":
        return MRPC_Dataset(config=config)
    if config.data.name == "openorca":
        return OpenOrcaDataset(config=config)
    if config.data.name == "hellaswag":
        return HellaSwagDataset(config=config)
    if config.data.name == "papillon":
        return PapillonDataset(config=config)
    msg = f"Unknown dataset name: {config.data.name}"
    raise ValueError(msg)


def get_data_loader(
    dataset: BaseDataset,
    batch_size: int | None = None,
    shuffle: bool = True,
) -> torch.utils.data.DataLoader:
    """Get the data loader based on the configuration."""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
        num_workers=1,
    )
