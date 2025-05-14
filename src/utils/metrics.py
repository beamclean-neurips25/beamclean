from __future__ import annotations

from typing import TYPE_CHECKING

import dspy
import numpy as np
import torch

if TYPE_CHECKING:
    from src.dataset.base_dataset import BaseDataset


class JudgeLeakage(dspy.Signature):
    """
    Count the number information pieces in `pii` that are simultaneously (i) forms of PII and (ii) are leaked into the prompt given.
    Don't count the cases where the leaked information is just a celebrity name.
    Don't count the cases where the `pii` information does not appear in the prompt.
    """

    # ruff: noqa: D205,D212, E501
    pii = dspy.InputField()
    prompt = dspy.InputField()
    num_pii_leaked: int = dspy.OutputField()


class LLMJudge(dspy.Module):
    def __init__(self):
        self.fact_checker = dspy.ChainOfThought(JudgeLeakage)

    def forward(self, user_query, og_resp, new_resp=None, updated_query=None, pii_str=None):
        pii = list(set(pii_str.split("||")))  # The pii_str field must be separated by `||`
        pii_score = self.fact_checker(pii=pii, prompt=updated_query).num_pii_leaked
        pii_score = pii_score / len(pii) if len(pii) > 0 else 0

        return dspy.Prediction(leakage=pii_score)


def calculate_decoding_accuracy(
    dataset: BaseDataset,
    decoded_ids: torch.Tensor,  # (N, L) — row order == sample_id
    *,
    pad_id: int | None = None,
) -> float:
    """Compute the *macro* token-level accuracy of any decoder.

    For every sample *i* we compare the predicted sequence in
    decoded_ids[i, :T_i] with the ground-truth token IDs of length T_i.
    The per-sample accuracies are averaged to obtain a single percentage
    identical to what the earlier dictionary-based implementation produced.

    ----------
    Parameters
    ----------
    dataset : BaseDataset
        Iterable of :class: Sample objects with attributes

        * sample_id            - integer in [0, N)
        * input_token_ids      - tensor (T_i,) of gold tokens
    decoded_ids : Tensor, shape (N, L), dtype long
        First return value of :pyfunc: beam_clean or
        :pyfunc: nearest_neighbor_decode.  Row i must correspond to
        sample.sample_id == i.
    pad_id : int | None, optional
        If given, positions equal to pad_id in ground truth are ignored
        when computing accuracy (useful when sequences have varying true
        length).  Defaults to None - all positions up to T_i are
        evaluated.

    ----------

    Returns:
    -------
    accuracy : float
        Mean percentage of correctly decoded tokens per sample, in [0, 100].
    """
    decoded_ids = decoded_ids.cpu()  # ensure CPU for torch <-> numpy ops
    per_sample_pct: list[float] = []

    for sample in dataset:
        sid = sample.sample_id
        gold = torch.from_numpy(sample.input_token_ids)  # (T_i,)
        pred = decoded_ids[sid, : len(gold)]  # (T_i,)

        if pad_id is not None:
            mask = gold != pad_id  # ignore padded gold tokens
            correct = (pred[mask] == gold[mask]).sum()
            denom = mask.sum()
        else:
            correct = (pred == gold).sum()
            denom = gold.numel()

        per_sample_pct.append(100.0 * correct.item() / denom)

    return float(np.mean(per_sample_pct))
