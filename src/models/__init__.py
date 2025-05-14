"""Transformer Encoder model code.

This code is obtained from the following link:
https://github.com/guocheng18/Transformer-Encoder/tree/master
"""

from __future__ import annotations

from src.models.utils import (
    BASE_MODEL_CHOICES,
    BASE_PRECISION_CHOICES,
    get_prior_model,
    get_surrogate_model,
    get_target_model,
)

__all__ = [
    "BASE_MODEL_CHOICES",
    "BASE_PRECISION_CHOICES",
    "get_prior_model",
    "get_surrogate_model",
    "get_target_model",
]
