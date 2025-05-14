from __future__ import annotations

import logging
import pathlib
import pickle
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import Dataset

from src.dataset.utils import load_noise_mechanism

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from src.dataset.sample import Sample


class BaseDataset(Dataset):
    """Dataset for BeamClean."""

    PAD_TOKEN_ID: int = 0

    def __init__(
        self,
        config: DictConfig,
    ):
        """Dataset Class to handle BeamClean.

        Args:
            config: Configuration object with data parameters.
        """
        self.config = config
        self.max_seq_len = config.data.truncated_seq_len
        self._sequences: list[Sample] = self.read_data()
        self.noise_fn = load_noise_mechanism(config.privacy.const_noise_params)
        self.post_process()

    def post_process(self) -> None:
        """Post-process the dataset after reading the data."""
        # TODO: This functionality should be outside the dataset
        if (
            self.config.privacy.noisy_embeddings_path is not None
            and self.config.privacy.load_noisy_embeddings
        ):
            logging.info(
                "Loading noisy embeddings from %s",
                self.config.privacy.noisy_embeddings_path,
            )
            self.load_noisy_embeddings(self.config.privacy.noisy_embeddings_path)

        elif self.config.privacy.noise_type == "constant":
            self._add_constant_noise()

        # Truncate the sequences to the max length
        for sample in self._sequences:
            if sample.embeddings.shape[0] > self.max_seq_len:
                sample.embeddings = sample.embeddings[: self.max_seq_len]
                sample.noisy_embeddings = sample.noisy_embeddings[: self.max_seq_len]
                if sample.input_token_ids is not None:
                    sample.input_token_ids = sample.input_token_ids[: self.max_seq_len]
                sample.metadata["truncated"] = True

    def read_data(self) -> list[dict]:
        """Read the data from the source and return a list of samples.

        Returns:
            list[dict]: A list of samples, where each sample is a dictionary.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> Sample:
        return self._sequences[idx]

    def _add_constant_noise(
        self,
    ) -> None:
        """Internal helper to add noise using a given sampling function."""
        logging.info("Adding noise to the embeddings...")
        for sample in self._sequences:
            sample.perturb_embeddings(self.noise_fn)

    def map_noisy_embeddings_to_closest(
        self,
        embedding_table: torch.Tensor,
    ) -> None:
        """Map noisy embeddings to the closest embedding in the table."""
        logging.info("Mapping noisy embeddings to the closest embedding...")
        for sample in self._sequences:
            sample.noisy_embeddings = self._map_to_closest(sample.noisy_embeddings, embedding_table)

    @staticmethod
    def _map_to_closest(
        noisy_embeddings: np.ndarray,
        embedding_table: torch.Tensor,
    ) -> torch.Tensor:
        """Map noisy embeddings to the closest embedding in the table."""
        # Compute the L2 distance between the noisy embeddings and the embedding table
        noisy_embeddings: torch.Tensor = torch.from_numpy(noisy_embeddings).float()
        noisy_embeddings = noisy_embeddings.to(embedding_table.device)
        distances = torch.cdist(noisy_embeddings, embedding_table)
        # Get the index of the closest embedding
        closest_idx = torch.argmin(distances, dim=1)
        # Map to the closest embedding
        return embedding_table[closest_idx]

    def save_noisy_embeddings(self, noisy_embeddings_path: str | pathlib.Path | None) -> None:
        # # Save the generated noisy embeddings to disk
        # result_path = pathlib.Path(self.config.run_dir)
        # noisy_embedding_path = result_path / "noisy_embeddings"
        with noisy_embeddings_path.open("wb") as f:
            pickle.dump([s.noisy_embeddings for s in self._sequences], f)

    def load_noisy_embeddings(self, noisy_embeddings_path: str | pathlib.Path | None) -> None:
        """Load constant noise from a file and add it to the embeddings."""
        if noisy_embeddings_path is None:
            noisy_embeddings_path = pathlib.Path(self.config.run_dir) / "noisy_embeddings"

        if isinstance(noisy_embeddings_path, str):
            noisy_embeddings_path = pathlib.Path(noisy_embeddings_path)

        if not noisy_embeddings_path.exists():
            logging.error(
                "No noisy embeddings found at %s. Skipping loading.",
                noisy_embeddings_path,
            )
            msg = f"No noisy embeddings found at {noisy_embeddings_path}"
            raise FileNotFoundError(msg)

        with noisy_embeddings_path.open("rb") as f:
            noisy_embeddings = pickle.load(f)

        for sample, noisy_embedding in zip(self._sequences, noisy_embeddings, strict=False):
            sample.noisy_embeddings = noisy_embedding

    def revert_to_original_noisy_embeddings(self) -> None:
        """Revert noisy embeddings to their original values."""
        for sample in self._sequences:
            if "original_noisy_embeddings" in sample:
                sample.noisy_embeddings = sample.original_noisy_embeddings
                # del sample.original_noisy_embeddings

    def get_all_sample_ids(
        self,
    ) -> tuple[int]:
        """Get all sample IDs in the dataset."""
        return tuple(sample.sample_id for sample in self._sequences)

    @staticmethod
    def collate_fn(batch: list[Sample]) -> dict:
        """Collate function to stack samples into a batch."""
        # 1. Find max sequence length in this batch ---
        seq_lens: list[int] = [item.embeddings.shape[0] for item in batch]
        embed_dim = batch[0].embeddings.shape[-1]

        max_seq_len = max(seq_lens)
        padded_embeddings = []
        padded_noisy_embeddings = []
        padded_token_ids = []
        padding_masks = []

        for item in batch:
            emb_np = item.embeddings  # NumPy array, shape [seq_len, embed_dim]
            noisy_emb_np = item.noisy_embeddings
            token_ids_np = item.input_token_ids  # or None

            # Convert to tensors here, on the CPU
            emb = torch.from_numpy(emb_np).float()
            noisy_emb = torch.from_numpy(noisy_emb_np).float()
            token_ids = torch.from_numpy(token_ids_np).long() if token_ids_np is not None else None

            seq_len = emb.shape[0]

            # Pad
            padded_emb = torch.zeros(max_seq_len, embed_dim, dtype=emb.dtype)
            padded_emb[:seq_len] = emb

            padded_noisy_emb = torch.zeros(max_seq_len, embed_dim, dtype=noisy_emb.dtype)
            padded_noisy_emb[:seq_len] = noisy_emb

            if token_ids is not None:
                padded_ids = torch.zeros(max_seq_len, dtype=token_ids.dtype)
                padded_ids[:seq_len] = token_ids
            else:
                padded_ids = None

            pad_mask = torch.zeros(max_seq_len, dtype=torch.long)
            pad_mask[:seq_len] = 1

            padded_embeddings.append(padded_emb)
            padded_noisy_embeddings.append(padded_noisy_emb)
            padded_token_ids.append(padded_ids)
            padding_masks.append(pad_mask)

        # Stack into final tensors
        padded_embeddings = torch.stack(
            padded_embeddings, dim=0
        )  # [B, truncated_seq_len, embed_dim]
        padded_noisy_embeddings = torch.stack(padded_noisy_embeddings, dim=0)
        if all(t is not None for t in padded_token_ids):
            padded_token_ids = torch.stack(padded_token_ids, dim=0)
        else:
            padded_token_ids = None
        padding_masks = torch.stack(padding_masks, dim=0)

        # Optionally collect metadata
        sample_ids = [item.sample_id for item in batch]
        metadata = [item.metadata for item in batch]

        return {
            "embeddings": padded_embeddings,
            "noisy_embeddings": padded_noisy_embeddings,
            "input_token_ids": padded_token_ids,
            "padding_mask": padding_masks,
            "sample_ids": sample_ids,
            "metadata": metadata,
            "seq_lens": seq_lens,
            "embed_dim": embed_dim,
            "max_seq_len": max_seq_len,
        }
