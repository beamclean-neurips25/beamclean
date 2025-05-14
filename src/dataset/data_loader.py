from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.utils.enums import NoiseFunction

if TYPE_CHECKING:
    from collections.abc import Callable

    import omegaconf


class UnknownNoiseFunctionError(ValueError):
    """Raised when an unknown noise function is specified."""


def generate_noisy_embeddings(
    original_embeddings: np.ndarray,
    noise_function: Callable,
    seed: int,
) -> np.ndarray:
    """Generate noisy embeddings based on the noise function."""
    rng = np.random.default_rng(seed)
    true_mean, true_variance = noise_function(original_embeddings)
    noise = rng.normal(true_mean, np.sqrt(true_variance), original_embeddings.shape)
    return original_embeddings + noise


def get_noise_function(config: omegaconf.DictConfig) -> Callable:
    """Get noise function based on the configuration."""
    noise_function = NoiseFunction(config.privacy.noise_type)
    if noise_function == NoiseFunction.CONSTANT:
        return lambda _: (
            config.privacy.const_noise_params.true_mean,
            config.privacy.const_noise_params.true_variance,
        )

    raise UnknownNoiseFunctionError(noise_function)


def generate_original_embeddings(
    prior_data: np.ndarray, num_data_points: int, seed: int
) -> np.ndarray:
    """Generate original embeddings from prior data."""
    rng = np.random.default_rng(seed)
    num_data_points = min(num_data_points, prior_data.shape[0])
    indices = rng.choice(prior_data.shape[0], num_data_points, replace=False)
    return prior_data[indices]


def load_experiment_data(
    config: dict,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Load prior data, original embeddings, and noisy embeddings.

    Args:
        config: Experiment configuration.
    """
    prior_data: np.ndarray = np.load(config.data.prior_data_path)
    embedding_data: np.ndarray = (
        np.load(config.data.embeddings_path) if config.data.embeddings_path else None
    )
    noisy_embedding_data: np.ndarray = (
        np.load(config.privacy.noisy_embeddings_path)
        if config.privacy.noisy_embeddings_path
        else None
    )
    return prior_data, embedding_data, noisy_embedding_data
