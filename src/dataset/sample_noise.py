from __future__ import annotations

import math

import torch
from torch.distributions.gamma import Gamma


def sample_noise_Chi(d_shape, eta):
    n_dim = d_shape[-1]

    # Sample magnitude from Gamma(d, 1/eta)
    gamma_dist = Gamma(torch.full(d_shape, n_dim), torch.full(d_shape, 1 / eta))
    l_lst = gamma_dist.sample()

    # Sample direction uniformly from the unit sphere using Gaussian method
    v_lst = torch.randn(d_shape)  # Sample from standard normal
    v_lst = v_lst / torch.norm(
        v_lst, dim=-1, keepdim=True
    )  # Normalize to unit norm

    # Scale unit vector by sampled magnitude
    return l_lst * v_lst


def chiDP_probability_density(z, eta):
    """Computes the probability density function (PDF) of ChiDP noise at a given point z.

    Args:
        z (torch.Tensor): The point(s) at which to evaluate the probability density.
        eta (float): The parameter controlling the noise scale.

    Returns:
        torch.Tensor: The probability density value at z.
    """
    d = z.shape[-1]  # Dimensionality of the space
    norm_z = torch.norm(z, dim=-1)  # Compute ||z||

    # Gamma distribution with shape = d, scale = 1/eta
    gamma_dist = Gamma(
        torch.tensor(d, dtype=torch.float32),
        torch.tensor(1 / eta, dtype=torch.float32),
    )

    # Compute Gamma PDF at ||z||
    gamma_pdf = gamma_dist.log_prob(norm_z).exp()

    # Volume of the unit ball in d dimensions
    unit_ball_volume = (math.pi ** (d / 2)) / torch.exp(
        torch.lgamma(torch.tensor(d / 2 + 1))
    )

    # Compute final probability density
    return gamma_pdf / unit_ball_volume
