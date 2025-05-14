"""Utility functions for BeamClean."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from src.dataset import get_dataset
from src.models import get_target_model

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from src.dataset.base_dataset import BaseDataset


def setup_environment(config: DictConfig) -> None:
    """Setup logging and random seeds."""
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(config.seed)
    np.random.default_rng(config.seed)


def setup_device() -> torch.device:
    """Setup compute device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_data_and_models(config: DictConfig) -> tuple[BaseDataset, Any, Any]:
    """Setup dataset and models."""
    dataset = get_dataset(config)
    tokenizer, target_model = get_target_model(config)
    return dataset, tokenizer, target_model


def log_memory_usage() -> float:
    """Log a more complete snapshot of GPU memory."""
    if not torch.cuda.is_available():
        logging.info("CUDA not available.")
        return None

    torch.cuda.synchronize()  # finish lazy allocations
    device = torch.cuda.current_device()

    free_driver, total_driver = torch.cuda.mem_get_info(device)  # raw driver info (bytes)
    used_driver = (total_driver - free_driver) / 1024**2
    # Return the percentage of memory used
    return used_driver / (total_driver / 1024**2)


def get_tensor_memory_usage(tensor: torch.Tensor) -> str:
    """Calculate memory usage of a tensor in a human-readable format.

    Args:
        tensor: The tensor to measure

    Returns:
        String with memory usage in appropriate units (B, KB, MB, GB)
    """
    bytes_size = tensor.element_size() * tensor.nelement()
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.2f} TB"
