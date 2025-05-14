"""BeamClean: A beam search decoding algorithm with surrogate model for noisy embeddings."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tqdm import tqdm

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

NN_NORM = 2


def nearest_neighbor_decode(
    dataloader: DataLoader,
    embedding_table: torch.Tensor,
    vocab_token_ids: torch.Tensor,
    *,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """1-NN decoding baseline (BeamClean §4.1).

    For every sample i and every time-step t we choose the vocabulary token
    whose embedding is nearest in Euclidean distance to the noisy embedding
    at that position. No language-model prior is used.

    Args:
        dataloader: DataLoader containing the dataset
        embedding_table: Token embedding table [V, d]
        vocab_token_ids: Vocabulary token IDs
        pad_id: Padding token ID

    Returns:
        Tuple containing:
            - Decoded token IDs [num_samples, max_len]
            - Negative distances [num_samples]
    """
    device = embedding_table.device
    dtype = embedding_table.dtype
    vocab_token_ids = vocab_token_ids.to(device)

    dataset = dataloader.dataset
    num_samples = len(dataset)
    max_len = max(len(s.input_token_ids) for s in dataset)

    decoded_ids = torch.full((num_samples, max_len), pad_id, dtype=torch.long)
    neg_dists = torch.empty(num_samples, dtype=torch.float32)

    # Process in batches
    for batch in tqdm(dataloader, desc="1-NN decoding", total=len(dataloader)):
        batch_noisy = batch["noisy_embeddings"].to(device=device, dtype=dtype)
        sample_ids = batch["sample_ids"]

        # Process batch
        distances = torch.cdist(batch_noisy, embedding_table, p=NN_NORM)
        min_distances, min_indices = distances.min(dim=2)

        # Update results
        for batch_idx, sample_id in enumerate(sample_ids):
            sequence_len = len(dataset[sample_id].input_token_ids)
            decoded_ids[sample_id, :sequence_len] = min_indices[batch_idx, :sequence_len].cpu()
            neg_dists[sample_id] = -min_distances[batch_idx, :sequence_len].sum().cpu()

        # Clear cache
        torch.cuda.empty_cache()

    return decoded_ids, neg_dists
