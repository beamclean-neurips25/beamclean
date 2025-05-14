from __future__ import annotations

import numpy as np


def generate_gaussian_noise(data: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Apply independent Gaussian noise to the given data.

    Args:
        data: The NumPy array to which Gaussian noise will be added.

    Returns:
        A new NumPy array with added Gaussian noise.
    """
    return np.random.normal(loc=mean, scale=std, size=data.shape)


def apply_mv_gaussian_noise(data: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Apply multivariate Gaussian noise to the given data.

    If `data` is 1D (shape `(d,)`), a single noise sample is generated.
    If `data` is 2D (shape `(n, d)`), `n` noise samples are generated.

    Args:
        data: A NumPy array of shape `(d,)` or `(n, d)`.

    Returns:
        A new NumPy array with added multivariate Gaussian noise.

    Raises:
        ValueError: If the dimensions of `data`, `mean`, or `cov` do not match.
    """
    if data.ndim == 1:
        d = data.shape[0]
        if mean.shape[0] != d or cov.shape != (d, d):
            raise ValueError(
                f"Shape mismatch: data={data.shape}, mean={mean.shape}, cov={cov.shape}"
            )
        noise = np.random.multivariate_normal(mean, cov)
    elif data.ndim == 2:
        n, d = data.shape
        if mean.shape[0] != d or cov.shape != (d, d):
            raise ValueError(
                f"Shape mismatch: data={data.shape}, mean={mean.shape}, cov={cov.shape}"
            )
        noise = np.random.multivariate_normal(mean, cov, size=n)
    else:
        raise ValueError("Data must be 1D or 2D.")
    return noise


def generate_laplace_noise(data: np.ndarray, mean: float, scale: float) -> np.ndarray:
    """Apply independent Laplacian noise to the given data.

    Args:
        data: The NumPy array to which Laplacian noise will be added.

    Returns:
        A new NumPy array with added Laplacian noise.
    """
    return np.random.laplace(loc=mean, scale=scale, size=data.shape)


def apply_mv_laplace_noise(data: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Apply multivariate Laplacian noise to the given data.

    This method uses a scale mixture approach:
        1. Sample an exponential random variable E (or a vector of them for multiple samples).
        2. Sample Z from a multivariate normal with zero mean and covariance matrix `cov`.
        3. Compute noise as: noise = sqrt(2 * E) * Z.
        4. Add the provided mean to the noise.

    Args:
        data: A NumPy array of shape `(d,)` or `(n, d)` representing the original data.

    Returns:
        A new NumPy array with added multivariate Laplacian noise.

    Raises:
        ValueError: If the dimensions of `data`, `mean`, or `cov` do not match.
    """
    if data.ndim == 1:
        d = data.shape[0]
        if mean.shape[0] != d or cov.shape != (d, d):
            raise ValueError(
                f"Shape mismatch: data={data.shape}, mean={mean.shape}, cov={cov.shape}"
            )
        E = np.random.exponential(scale=1.0)
        Z = np.random.multivariate_normal(np.zeros(d), cov)
        noise = np.sqrt(2 * E) * Z
    elif data.ndim == 2:
        n, d = data.shape
        if mean.shape[0] != d or cov.shape != (d, d):
            raise ValueError(
                f"Shape mismatch: data={data.shape}, mean={mean.shape}, cov={cov.shape}"
            )
        E = np.random.exponential(scale=1.0, size=n)
        # Generate n independent multivariate normals
        Z = np.array([np.random.multivariate_normal(np.zeros(d), cov) for _ in range(n)])
        noise = np.sqrt(2 * E)[:, None] * Z
    else:
        raise ValueError("Data must be 1D or 2D.")
    return mean + noise


# def generate_isotropic_laplace_noise(data: np.ndarray, eta: float) -> np.ndarray:
#     """Apply isotropic Laplacian noise to the given data.

#     Args:
#         data: The NumPy array to which Laplacian noise will be added.

#     Returns:
#         A new NumPy array with added isotropic Laplacian noise.
#     """
#     d_shape = data.shape
#     n_dim = d_shape[-1]
#     alpha = torch.ones(d_shape) * n_dim
#     beta = torch.ones(d_shape) * 1 / eta
#     m = Gamma(alpha, beta)
#     l_lst = m.sample()
#     # v_lst = -2 * torch.rand(d_shape) + 1
#     v_lst = torch.randn(d_shape)
#     noise: torch.Tensor = l_lst * v_lst
#     return noise.numpy()


def generate_isotropic_laplace_noise(
    data: np.ndarray, scale: float
) -> np.ndarray:  # d: int, Delta: float, eps: float
    """Draw a sample from the d-dimensional spherical Laplace distribution
    with parameters (Delta, eps).  Equivalently, this is the distribution
    of noise kappa ~ exp(- (eps/Delta)*||kappa||_2).

    The procedure is:
      1) Sample a random direction uniformly on the unit sphere in R^d.
      2) Sample a radius from Gamma(d, scale=Delta/eps).
      3) Multiply the direction by that radius.

    Args:
        d: Dimension.
        Delta: The L2-sensitivity parameter (Δ).
        eps: The privacy parameter (ε).

    Returns:
        A NumPy vector (shape (d,)) drawn from the spherical Laplace distribution.
    """
    # Step 1: Random direction in R^d (uniform on the unit sphere).
    x = np.random.normal(0, 1, data.shape)
    norm_x = np.linalg.norm(x, 2, axis=-1, keepdims=True)
    # Normalize to get a unit vector
    direction = x / norm_x

    # Step 2: Sample the magnitude from Gamma(d, scale=Delta/eps).
    radius = np.random.gamma(shape=data.shape[-1], scale=scale, size=data.shape)

    # Step 3: Final noise vector.
    return radius * direction


# TODO: Ask this, so this is an edge case where
# I need to set the perturb embedding which will be break the generalization of the code.
def apply_custom_noise(data: np.ndarray, _perturb_arr: np.ndarray) -> np.ndarray:
    """Apply the custom perturbation to the given data.

    Args:
        data: The NumPy array to which the custom perturbation will be added.
        perturb_arr: A NumPy array representing the custom perturbation
            to be added to the data. Must broadcast to `data.shape`.

    Returns:
        A new NumPy array containing the data with the custom perturbation added.
    """
    if _perturb_arr is None:
        raise ValueError("Perturbation array must be set before applying custom noise.")
    # Ensure that the perturbation array has the same shape as data
    # or can be broadcast to that shape
    assert _perturb_arr is not None, "Perturbation array must be set."
    assert _perturb_arr.shape == data.shape
    perturbed_data = data + _perturb_arr
    _set_perturb_arr_none()  # Reset perturbation after use
    return perturbed_data


def set_perturb_arr(perturb_arr: np.ndarray):
    """Set the perturbation array for the custom noise.

    Args:
        perturb_arr: A NumPy array representing the custom perturbation
            to be added to the data. Must broadcast to `data.shape`.
    """
    _perturb_arr = perturb_arr
    assert _perturb_arr is not None, "Perturbation array must be set before applying custom noise."


def _set_perturb_arr_none():
    """Set the perturbation array to None."""
    _perturb_arr = None
