from __future__ import annotations

from collections.abc import Callable
from functools import partial

import omegaconf
import omegaconf.omegaconf

from src.dataset.noise_mechanisms import (
    generate_gaussian_noise,
    generate_isotropic_laplace_noise,
    generate_laplace_noise,
)


def load_noise_mechanism(config: omegaconf.omegaconf.DictConfig) -> Callable:
    """Load a noise mechanism based on the provided configuration.

    Args:
        config: A dictionary containing the configuration for the noise mechanism.

    Returns:
        A callable that applies the specified noise mechanism.
    """
    noise_type = config.get("noise_dist", "gaussian")
    if noise_type == "gaussian":
        gaussian_noise_params: omegaconf.omegaconf.DictConfig = config.get("gaussian", {})
        return partial(
            generate_gaussian_noise,
            mean=gaussian_noise_params.get("mean", 0.0),
            std=gaussian_noise_params.get("std", 1.0),
        )
    if noise_type == "l1_laplace":
        laplace_noise_params: omegaconf.omegaconf.DictConfig = config.get("l1_laplace", {})
        return partial(
            generate_laplace_noise,
            mean=laplace_noise_params.get("mean", 0.0),
            scale=laplace_noise_params.get("scale", 1.0),
        )
    if noise_type == "l2_laplace":
        isotropic_laplace_params: omegaconf.omegaconf.DictConfig = config.get("l2_laplace", {})
        return partial(
            generate_isotropic_laplace_noise,
            scale=isotropic_laplace_params.get("scale", 0.0),
        )
    if noise_type == "gaussian_with_mean":
        gaussian_with_mean_params: omegaconf.omegaconf.DictConfig = config.get(
            "gaussian_with_mean", {}
        )
        return partial(
            generate_gaussian_noise,
            mean=gaussian_with_mean_params.get("mean", 0.0),
            std=gaussian_with_mean_params.get("std", 1.0),
        )

    error_msg = f"Unknown noise type: {noise_type}"
    raise ValueError(error_msg)
