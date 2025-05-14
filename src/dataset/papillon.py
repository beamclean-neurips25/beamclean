from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from src.dataset.base_dataset import BaseDataset
from src.dataset.sample import Sample
from src.models import get_target_model

if TYPE_CHECKING:
    from omegaconf import DictConfig


class PapillonDataset(BaseDataset):
    """Papillon dataset class."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config=config)

    def read_data(self) -> list[Sample]:
        """Read the Papillon dataset and return a list of samples.

        Returns:
            List of samples.
        """
        tokenizer, model = get_target_model(self.config)
        embedding_table = model.get_input_embeddings()

        dataset = load_dataset("Columbia-NLP/PUPA", "pupa_new")["train"]
        rng = np.random.default_rng(22)
        # Choose random 5 numbers between 0 and 1000
        if self.config.data.subset_size is not None:
            logging.info("Using sample ids len %s", self.config.data.subset_size)

            dataset_sample_ids = rng.choice(
                len(dataset), self.config.data.subset_size, replace=False
            )
            _iter = dataset.select(dataset_sample_ids)
        else:
            _iter = dataset

        sequences = []

        for idx, sample in tqdm(enumerate(_iter)):
            complete = sample["user_query"]
            pii_str: str = sample["pii_units"]
            sentence_tokenized: list = tokenizer.encode(complete, add_special_tokens=False)

            # Truncate the sentences to the maximum length
            if len(sentence_tokenized) > self.config.data.truncated_seq_len:
                sentence_tokenized = sentence_tokenized[: self.config.data.truncated_seq_len]

            embeddings = (
                embedding_table(torch.tensor(sentence_tokenized)).squeeze(0).detach().cpu().numpy()
            )
            # Create a sample
            _sample = Sample(
                sample_id=idx,
                embeddings=embeddings,
                input_token_ids=np.array(sentence_tokenized),
                metadata={
                    "complete": complete,
                    "pii_str": pii_str,
                },
            )
            sequences.append(_sample)
        return sequences
