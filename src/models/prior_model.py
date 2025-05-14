from typing import Any

import torch

BOS_TOKEN_ID = 128000


class PriorModel:
    """Wrapper class to handle the prior model."""

    def __init__(self, model: Any, device: torch.device) -> None:
        self.model = model.to(device)
        self.device = device

    def get_embedding_table(self) -> torch.Tensor:
        """Return the embedding table of the model.

        Note: This method is used to extract the embedding table of the model.
        The model should be a PreTrainedModel from the transformers library.
        If the model does not have a get_input_embeddings method, an error is raised.

        Returns:
            torch.Tensor: Embedding table of the model, with a shape of (vocab_size, embedding_dim).
        """
        # Check if the model has a get_input_embeddings method
        if hasattr(self.model, "get_input_embeddings"):
            return self.model.get_input_embeddings().weight

        raise NotImplementedError("Model does not have a get_input_embeddings method")

    def calculate_log_probs(
        self,
        candidate_tokens: torch.Tensor,
        beam_sequences: torch.Tensor | None,
        batch_size: int,
    ) -> torch.Tensor:
        """Calculate the probabilities of the candidate tokens using the model.

        Args:
            candidate_tokens: The candidate tokens for which the loss needs to be calculated.
            beam_sequences: The beam sequences for the current time step.
            batch_size: The batch size of the input.

        Returns:
            torch.Tensor: The log probabilities of the candidate tokens.
        """
        # Calculate the prior log-likelihood for the sequences
        prior_probs = self._prob_fn(
            beam_sequences=beam_sequences,
            batch_size=batch_size,
        )
        prior_log_probs_for_candidates = torch.log(prior_probs[:, -1, :])

        # Calculate the negative log-likelihood
        return prior_log_probs_for_candidates[:, candidate_tokens]

    def _prob_fn(
        self,
        beam_sequences: torch.Tensor,
        batch_size: int,
    ):
        # Add BOS token to the start of the each sequence in the token_ids
        # which is a tensor of shape [num_possible_seq, seq_len]
        if beam_sequences is not None:
            beam_sequences = torch.cat(
                [
                    torch.full(
                        (beam_sequences.size(0), 1),
                        BOS_TOKEN_ID,
                        device=self.device,
                    ),
                    beam_sequences,
                ],
                dim=1,
            )
        else:
            beam_sequences = torch.full(
                (batch_size, 1), BOS_TOKEN_ID, device=self.device
            )

        attention_mask = torch.ones_like(beam_sequences)

        with torch.no_grad():
            output = self.model(
                input_ids=beam_sequences,
                attention_mask=attention_mask,
            )
            # outputs.logits -> [batch_size, seq_len, vocab_size]
            # If you have a convenience function to compute average or sum logprob:
            # e.g. negative log-likelihood
            # This is up to your model's interface
        # Shape batch_size
        return torch.softmax(
            output.logits, dim=-1
        )  # shape: batch_size x sequence_length x vocab_size
