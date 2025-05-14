"""BeamClean - Scoring module.

-------------------------------------------------
All shape checks are now expressed through jaxtyping annotations plus
the optional runtime hook.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:  # avoid circular import at run-time
    from collections.abc import Callable

    from jaxtyping import Float, Int


class ScoringAlgorithm(ABC):
    """Interface for any beam-search scoring rule."""

    @abstractmethod
    def compute_scores(
        self,
        noisy_embeddings: Float[Tensor, "batch embedding_dim"],
        beam_ids: Int[Tensor, "batch beam time"],
        vocab_token_ids: torch.Tensor,
        *,
        vocab_chunk_size: int = 4_096,
    ) -> Float[Tensor, "batch beam vocab"]: ...


class JointProbabilityScoring(ScoringAlgorithm):
    """Log p  =  log p_surrogate  +  log p_prior."""

    def __init__(
        self,
        surrogate_logp_fn: Callable[
            [Float[Tensor, "batch embedding_dim"], dict[str, Tensor]],
            Float[Tensor, "batch beam chunk"],
        ],
        prior_logp_fn: Callable[
            [Int[Tensor, "batch beam chunk time"], Int[Tensor, "batch beam chunk time_plus_1"]],
            Float[Tensor, "batch beam chunk"],
        ],
        embedding_table: Float[Tensor, "vocab embedding_dim"],  # Torch embedding matrix
        device: torch.device,
    ) -> None:
        self.surrogate_logp_fn = surrogate_logp_fn
        self.prior_logp_fn = prior_logp_fn
        self.embedding_table = embedding_table
        self.device = device

    # --------------------------------------------------------------------- #
    #  Helper: build candidate sequences for a vocab chunk                  #
    # --------------------------------------------------------------------- #
    def _get_candidate_sequences(
        self,
        vocab_token_ids: torch.Tensor,
        start: int,
        chunk: int,
        beam_ids: Int[Tensor, "batch beam time"],
    ) -> Int[Tensor, "batch beam chunk time_plus_1"]:
        """Append `chunk` vocab IDs to every beam prefix."""
        batch_size, beam_width, time_step = beam_ids.shape
        cand = vocab_token_ids[start : start + chunk]  # (C,)
        cand: Int[Tensor, "batch beam chunk 1"] = cand.view(1, 1, -1, 1).expand(
            batch_size, beam_width, -1, 1
        )  # (B,Z,C,1)

        if time_step == 0:
            return cand  # first step

        prefix: Int[Tensor, "batch beam chunk time"] = beam_ids.unsqueeze(2).expand(
            batch_size, beam_width, cand.size(2), time_step
        )
        return torch.cat([prefix, cand], dim=-1)  # (B,Z,C,t+1)

    # --------------------------------------------------------------------- #
    #  Helper: lookup embeddings for a batch of candidate sequences         #
    # --------------------------------------------------------------------- #
    def _embed(
        self,
        sequences: Int[Tensor, "batch beam chunk time"],
    ) -> Float[Tensor, "batch beam chunk embedding_dim"]:
        flat = sequences.view(-1, sequences.size(-1))  # (batch_size*beam_width*chunk_size, time)
        emb = self.embedding_table[flat]  # (batch_size*beam_width*chunk_size, embedding_dim)
        batch_size, beam_width, chunk_size, _ = sequences.shape
        return emb.view(
            batch_size, beam_width, chunk_size, -1
        )  # (batch_size, beam_width, chunk_size, embedding_dim)

    # --------------------------------------------------------------------- #
    #  Core API                                                             #
    # --------------------------------------------------------------------- #
    def compute_scores(
        self,
        noisy_embeddings: Float[Tensor, "batch embedding_dim"],
        beam_ids: Int[Tensor, "batch beam time"],
        vocab_token_ids: torch.Tensor,
        *,
        vocab_chunk_size: int = 2_048,
    ) -> Float[Tensor, "batch beam vocab_size"]:
        """Return joint log-probabilities for every vocab item."""
        vocab_size = vocab_token_ids.numel()
        logp_surr_parts: list[Float[Tensor, "batch beam chunk"]] = []
        with torch.no_grad():
            logp_prior: Float[Tensor, "batch beam vocab_size"] = self.prior_logp_fn(
                beam_ids
            )  # (B,Z,V)

        for start in range(0, vocab_size, vocab_chunk_size):
            chunk_size = min(vocab_chunk_size, vocab_size - start)
            cand_token_ids = vocab_token_ids[start : start + chunk_size]  # (C,)

            logp_surr: Float[Tensor, "batch chunk"] = self.surrogate_logp_fn(
                noisy_embeddings, cand_token_ids
            )  # (B,C)

            logp_surr_parts.append(logp_surr.unsqueeze(1))  # (B,Z,C)

        logp_surr = torch.cat(logp_surr_parts, dim=2)  # (B,Z,V)
        return logp_prior + logp_surr  # (B,Z,V)
