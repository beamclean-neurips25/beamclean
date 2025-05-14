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


class MRPC_Dataset(BaseDataset):
    """MRPC dataset class."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config=config)

    def read_data(self) -> list[Sample]:
        """Read the MRPC dataset and return a list of samples.

        Returns:
            List of samples.
        """
        tokenizer, model = get_target_model(self.config)
        embedding_table = model.get_input_embeddings()

        dataset = load_dataset("glue", "mrpc", split="train")
        rng = np.random.default_rng(22)
        # Choose random 5 numbers between 0 and 1000
        if self.config.data.sample_ids_to_use_path is not None:
            logging.info(
                "Using sample ids from %s", self.config.data.sample_ids_to_use_path
            )

            dataset_sample_ids = np.load(self.config.data.sample_ids_to_use_path)
        else:
            dataset_sample_ids = np.arange(len(dataset))

        if self.config.data.subset_size is not None:
            random_sample_indices = rng.choice(
                len(dataset_sample_ids), self.config.data.subset_size, replace=False
            )
            dataset_sample_ids = dataset_sample_ids[random_sample_indices]

        _iter = dataset.select(dataset_sample_ids)

        sequences = []

        for idx, sample in tqdm(enumerate(_iter)):
            sentence1 = sample["sentence1"]
            sentence2 = sample["sentence2"]
            complete = sentence1  # + sentence2
            sentence_tokenized: list = tokenizer.encode(
                complete, add_special_tokens=False
            )

            embeddings = (
                embedding_table(torch.tensor(sentence_tokenized))
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )
            # Create a sample
            sample = Sample(
                sample_id=idx,
                embeddings=embeddings,
                input_token_ids=np.array(sentence_tokenized),
                metadata={
                    "sentence1": sentence1,
                    "sentence2": sentence2,
                    "complete": complete,
                },
            )
            sequences.append(sample)

        return sequences
