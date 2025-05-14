from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.dataset.base_dataset import BaseDataset
from src.dataset.sample import Sample


class OpenOrcaDataset(BaseDataset):
    """OpenOrca dataset class.

    Expected columns in `df`:
    - 'sample_id': groups rows into sequences
    - 'sample_idx': (optional) ordering of rows within each sequence
    - 'embeddings': per-row (token) embedding, which we stack
    - 'noisy_embeddings': per-row embedding corrupted by SGT
    - Possibly 'input_token_ids': a list or scalar for each row
    - Other metadata columns (input_text, etc.)
    which might be duplicated across the group

    A Dataset that groups rows in a DataFrame by 'sample_id' so that each sample
    is one full sequence. Each row in the DataFrame typically represents
    one token (or chunk).
    We then stack them to form a single sequence of embeddings.
    """

    def __init__(self, config) -> None:
        super().__init__(config=config)

    def read_data(
        self,
    ) -> None:
        """Read the OpenOrca dataset and return a list of samples.

        df (pd.DataFrame): A DataFrame with the columns mentioned above.
        transform (callable, optional): If provided, a function that transforms
            the entire merged sample dict after stacking.
        group_col (str): Column name to group by (default 'sample_id').
        order_col (str or None): Column name to sort each group by
            (default 'sample_idx'). If None, no sorting is done.
        """
        # Load the DataFrame
        logging.info("Loading dataset from %s", self.config.data.parquet_file_path)
        parquet_file_path = self.config.data.parquet_file_path
        df = pd.read_parquet(parquet_file_path)

        # --- Group rows by sequence (sample_id) ---
        df["sample_idx"] = df.apply(
            lambda row: (row[self.config.data.batch_col] - 1)
            * df[self.config.data.batch_inner_col].nunique()
            + row[self.config.data.batch_inner_col],
            axis=1,
        )
        grouped = df.groupby("sample_idx", sort=False)
        _sequences = []

        truncated_seq_len = self.config.data.truncated_seq_len

        # --- Progress bar ---
        prog_bar = grouped

        # Log the maximum sequence length
        logging.info("Max sequence length: %d", truncated_seq_len)

        # --- Iterate through each group (sequence) ---
        logging.info("Loading dataset...")
        for seq_id, sequence_df in prog_bar:
            # Identify the non-padding rows
            non_padding_indices = sequence_df.apply(
                lambda row: not np.array_equal(row["embeddings"], row["noisy_embeddings"]),
                axis=1,
            )
            sequence_df_no_padding = sequence_df[non_padding_indices].reset_index(drop=True)

            # Also truncate the df to the max sequence length
            # (for the case where we have padding)
            sequence_df_no_padding = sequence_df_no_padding[:truncated_seq_len]
            # Store as NumPy arrays, not Tensors
            embeddings_np = np.stack(sequence_df_no_padding["embeddings"].values, axis=0)
            noisy_embeddings_np = np.stack(
                sequence_df_no_padding["noisy_embeddings"].values, axis=0
            )

            # If you have token IDs
            input_token_ids_np = None
            if "input_token_ids" in sequence_df_no_padding.columns:
                token_id_list = sequence_df_no_padding.loc[0, "input_token_ids"][
                    non_padding_indices
                ]
                token_id_list = token_id_list[:truncated_seq_len]
                input_token_ids_np = np.array(token_id_list, dtype=np.int64)

            # Grab some metadata
            first_row = sequence_df.iloc[0]
            input_text = first_row.get("input_text", None)
            _batch_idx = first_row.get("_batch_idx", None)

            sample = Sample(
                sample_id=seq_id,
                embeddings=embeddings_np,
                noisy_embeddings=noisy_embeddings_np,
                input_token_ids=input_token_ids_np,
                metadata={"input_text": input_text, "_batch_idx": _batch_idx},
            )

            _sequences.append(sample)

        return _sequences
