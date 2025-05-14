"""Surrogate models for BeamClean."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


class SurrogateModel(ABC, nn.Module):
    """Base class for surrogate models."""

    def __init__(self) -> None:
        """Initialize the surrogate model."""
        super().__init__()
        self.device = torch.device("cpu")

    def to(
        self,
        *args: torch.device | torch.dtype | torch.nn.Module,
        **kwargs: dict[str, bool | int | float | str],
    ) -> SurrogateModel:
        """Override 'to' to set device after parameters are created."""
        super().to(*args, **kwargs)
        # Get device from the first parameter
        self.device = next(self.parameters()).device
        return self

    @abstractmethod
    def forward(
        self,
        embeddings: Tensor,  # [B, Z, C, d]
    ) -> dict[str, Tensor]:
        """Forward pass of the surrogate model.

        Args:
            embeddings: Input embeddings [B, Z, C, d]

        Returns:
            Dictionary of model outputs (parameters for probability distribution)
        """

    @abstractmethod
    def log_pdf(
        self,
        noisy_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        noise_stats: dict,
    ) -> torch.Tensor:
        """Log probability density function of the surrogate model."""

    @abstractmethod
    def log(self) -> None:
        """Log the parameters of the surrogate model."""


class IsotropicGaussianSurrogate(SurrogateModel):
    """Isotropic Gaussian surrogate model."""

    def __init__(
        self,
        initial_std: float | None = None,
        use_checkpointing: bool = True,
    ) -> None:
        """Initialize isotropic Gaussian surrogate model.

        Args:
            initial_std: Initial standard deviation value
            use_checkpointing: Whether to use gradient checkpointing
        """
        super().__init__()
        self.use_checkpointing = use_checkpointing
        self.log_std = nn.Parameter(
            torch.log(torch.tensor((initial_std,)))
            if initial_std is not None
            else torch.log(torch.ones(1) * 0.3)
        )
        logging.info("Initial std: %.4f", torch.exp(self.log_std).item())

    def forward(
        self,
        embeddings: Tensor,  # [batch_size, beam_width, num_candidates, embedding_dim]
    ) -> dict[str, Tensor]:
        """Forward pass through the surrogate model.

        Args:
            embeddings: Input embeddings [batch_size, beam_width, num_candidates, embedding_dim]

        Returns:
            Dictionary containing:
                - log_std: Log standard deviation
        """
        return self._forward_impl(embeddings)

    def _forward_impl(
        self,
        *_: torch.Tensor,
    ) -> dict[str, Tensor]:
        """Implementation of forward pass.

        Returns:
            Dictionary containing:
                - std: Standard deviation
        """
        # For isotropic Gaussian, mean is the input embeddings
        # and std is the same for all dimensions
        return {"std": torch.exp(self.log_std)}

    def log_pdf(
        self,
        noisy_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        noise_stats: dict,
    ) -> torch.Tensor:
        """Gaussian PDF function for log likelihood calculation."""
        # Compute the residuals between the current embedding and each table entry.
        # Compute squared Euclidean distances.
        # Noisy Embeddings Shape: batch_size x embedding_dim
        # Candidate Embeddings Shape: batch_size x num_candidates x embedding_dim
        std = noise_stats.get("std")
        var = std**2
        residuals = noisy_embeddings.unsqueeze(1) - candidate_embeddings
        squared_distance = torch.sum(residuals**2, dim=-1)

        d = residuals.shape[-1]
        # Compute the normalization constant: - (d/2) * log(2 * pi * var)
        norm_const = -(d / 2) * torch.log(
            2.0 * torch.pi * var,
        )
        # Compute and return the log likelihood.
        return norm_const - squared_distance / (2.0 * var)

    def log(self) -> None:
        """Log the parameters of the surrogate model."""
        logging.info("std=%.4f", torch.exp(self.log_std).item())


class IsotropicGaussianSurrogateWithMean(SurrogateModel):
    """Isotropic Gaussian surrogate model with mean."""

    def __init__(
        self,
        initial_std: float | None = None,
        use_checkpointing: bool = True,
    ) -> None:
        """Initialize isotropic Gaussian surrogate model with mean.

        Args:
            initial_std: Initial standard deviation value
            use_checkpointing: Whether to use gradient checkpointing
        """
        super().__init__()
        self.use_checkpointing = use_checkpointing
        self.mean = nn.Parameter(torch.ones(1) * 0.0)
        self.log_std = nn.Parameter(
            torch.log(torch.tensor((initial_std,)))
            if initial_std is not None
            else torch.log(torch.ones(1) * 0.3)
        )
        logging.info("Initial std: %.4f", torch.exp(self.log_std).item())

    def forward(
        self,
        embeddings: Tensor,  # [batch_size, beam_width, num_candidates, embedding_dim]
    ) -> dict[str, Tensor]:
        """Forward pass through the surrogate model.

        Args:
            embeddings: Input embeddings [batch_size, beam_width, num_candidates, embedding_dim]

        Returns:
            Dictionary containing:
                - log_std: Log standard deviation
        """
        return self._forward_impl(embeddings)

    def _forward_impl(
        self,
        *_: torch.Tensor,
    ) -> dict[str, Tensor]:
        """Implementation of forward pass.

        Returns:
            Dictionary containing:
                - mean: Mean
                - std: Standard deviation
        """
        # For isotropic Gaussian, mean is the input embeddings
        # and std is the same for all dimensions
        return {"mean": self.mean, "std": torch.exp(self.log_std)}

    def log_pdf(
        self,
        noisy_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        noise_stats: dict,
    ) -> torch.Tensor:
        """Gaussian PDF function for log likelihood calculation."""
        # Compute the residuals between the current embedding and each table entry.
        # Compute squared Euclidean distances.
        # Noisy Embeddings Shape: batch_size x embedding_dim
        # Candidate Embeddings Shape: batch_size x num_candidates x embedding_dim
        mean = noise_stats.get("mean")
        std = noise_stats.get("std")
        var = std**2
        residuals = noisy_embeddings.unsqueeze(1) - candidate_embeddings
        residuals = residuals - mean
        squared_distance = torch.sum(residuals**2, dim=-1)

        d = residuals.shape[-1]
        # Compute the normalization constant: - (d/2) * log(2 * pi * var)
        norm_const = -(d / 2) * torch.log(
            2.0 * torch.pi * var,
        )
        # Compute and return the log likelihood.
        return norm_const - squared_distance / (2.0 * var)

    def log(self) -> None:
        """Log the parameters of the surrogate model."""
        logging.info("std=%.4f, mean=%.4f", torch.exp(self.log_std).item(), self.mean.item())


class IsotropicL1LaplacianSurrogate(SurrogateModel):
    """Isotropic L1 Laplacian surrogate model."""

    def __init__(
        self,
        initial_scale: float | None = None,
        use_checkpointing: bool = True,
    ) -> None:
        """Initialize isotropic L1 Laplacian surrogate model.

        Args:
            initial_scale: Initial scale value
            use_checkpointing: Whether to use gradient checkpointing
        """
        super().__init__()
        self.use_checkpointing = use_checkpointing
        self.log_scale = nn.Parameter(
            torch.log(torch.tensor((initial_scale,)))
            if initial_scale is not None
            else torch.log(torch.ones(1) * 0.0001)
        )

    def forward(
        self,
        embeddings: Tensor,  # [batch_size, beam_width, num_candidates, embedding_dim]
    ) -> dict[str, Tensor]:
        """Forward pass through the surrogate model.

        Args:
            embeddings: Input embeddings [batch_size, beam_width, num_candidates, embedding_dim]

        Returns:
            Dictionary containing:
                - log_scale: Log scale
        """
        return self._forward_impl(embeddings)

    def _forward_impl(
        self,
        *_: torch.Tensor,
    ) -> dict[str, Tensor]:
        """Implementation of forward pass.

        Returns:
            Dictionary containing:
                - scale: Scale
        """
        # For isotropic Gaussian, mean is the input embeddings
        # and std is the same for all dimensions
        return {"scale": torch.exp(self.log_scale)}

    def log_pdf(
        self,
        noisy_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        noise_stats: dict,
    ) -> torch.Tensor:
        """Laplace PDF function for log likelihood calculation using L1 distance."""
        # Compute the absolute differences (L1 distance) between the current embedding
        # and each embedding in the table.
        # current_embedding.unsqueeze(0) reshapes it to (1, dim) for broadcasting.
        scale = noise_stats.get("scale")
        residuals = torch.abs(noisy_embeddings.unsqueeze(1) - candidate_embeddings)

        # Sum the absolute differences over the embedding dimensions.
        abs_sum = torch.sum(residuals, dim=-1)

        # Get the dimensionality of the embeddings.
        dim = residuals.shape[-1]

        # Calculate the normalization term.
        # Note: Using torch.tensor to keep the operation in the tensor domain.
        log_norm = -dim * torch.log(2.0 * scale)

        # Compute and return the log likelihood.
        return log_norm - abs_sum / scale

    def log(self) -> None:
        """Log the parameters of the surrogate model."""
        logging.info("scale=%.4f", torch.exp(self.log_scale).item())


class IsotropicL2LaplacianSurrogate(IsotropicL1LaplacianSurrogate):
    """Isotropic L2 Laplacian surrogate model."""

    def log_pdf(
        self,
        noisy_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        noise_stats: dict,
    ) -> torch.Tensor:
        """Laplace PDF function for log likelihood calculation using L2 distance."""
        scale = noise_stats.get("scale")
        emb_dim = candidate_embeddings.shape[-1]
        residuals = noisy_embeddings.unsqueeze(1) - candidate_embeddings
        # Take the norm of the residuals
        r = residuals.norm(p=2, dim=-1)
        log_c = -emb_dim * torch.log(scale)  # d ln b
        return log_c - r / scale


def process_surrogate_model_input_dependent(
    surrogate_model: torch.nn.Module,
    candidate_beam_embeddings: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Process candidate beam embeddings through the surrogate model.

    Extract the mean and standard deviation for the last time step.

    Args:
        surrogate_model (nn.Module): The surrogate model to process embeddings.
        candidate_beam_embeddings (torch.Tensor): The input embeddings for the surrogate model.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
        The mean and standard deviation tensors for the last time step.
    """
    # Forward pass through the surrogate model
    output: torch.Tensor = surrogate_model(candidate_beam_embeddings)
    mean, std = output.chunk(2, dim=-1)
    mean = mean[:, :, -1, :]  # Select the last time step
    std = std[:, :, -1, :]  # Select the last time step
    return mean, std
