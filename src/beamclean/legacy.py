from __future__ import annotations

import torch


def prior_score(
    input_token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    vocab_token_ids: torch.Tensor,
    prior_model: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """Compute log-probabilities of vocabulary tokens as predicted by a causal-LM prior.

    The function performs a forward pass through "prior_model" under
    "torch.no_grad()" to avoid gradient tracking.
    Only the logits corresponding to "vocab_token_ids" for the last input
    position are retained.
    A log-softmax is applied over that subset, yielding a matrix of shape
    (batch_size, |vocab_token_ids|).

    Args:
        input_token_ids:
            Tensor of shape (B, T) containing the prefix token IDs fed to the
            language model.
        attention_mask:
            Tensor of shape (B, T) with 1 for valid tokens and 0
            for padding, as expected by HuggingFace models.
        vocab_token_ids:
            Tensor of shape (V,) - the specific vocabulary IDs whose
            probabilities we care about (e.g., candidate set used by BeamClean).
        prior_model:
            A HuggingFace PreTrainedModel (e.g., GPT-2, LLaMA) that returns
            .logits of shape (B, T, |V|) for a causal language-model head.
        device:
            Device to use for computations (e.g., "cuda" or "cpu").

    Returns:
        log_probs:
            NumPy array of shape (B, V) where log_probs[i, j] is
            log p_prior(token = vocab_token_ids[j] | prefix = input_token_ids[i]).
    """
    input_token_ids = input_token_ids.to(device)
    attention_mask = attention_mask.to(device)
    vocab_token_ids = vocab_token_ids.to(device)
    with torch.no_grad():
        outputs = prior_model(input_ids=input_token_ids, attention_mask=attention_mask)

    logits = outputs.logits
    # Get the last token's logits
    last_token_logits = logits[:, -1, :][:, torch.tensor(vocab_token_ids).to(logits.device)]
    # probs = (
    #     torch.nn.functional.log_softmax(outputs.logits[:, -1, :], dim=-1).cpu().numpy()
    # )
    # return probs[:, subset_token_ids]
    # Apply softmax to get probabilities
    return torch.nn.functional.log_softmax(last_token_logits, dim=-1)
