from __future__ import annotations

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


class HellaSwagDataset(BaseDataset):
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

        dataset = load_dataset("Rowan/hellaswag", split="train")

        rng = np.random.default_rng(22)
        # Choose random 5 numbers between 0 and 1000
        random_sample_indices = rng.choice(1000, 100, replace=False)

        _iter = dataset.select(random_sample_indices)

        sequences = []

        for idx, sample in tqdm(enumerate(_iter)):
            complete = sample["ctx"]
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
                    "complete": complete,
                },
            )
            sequences.append(sample)

        return sequences
