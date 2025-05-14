"""BeamClean trainer module for beam-search decoding with surrogate models.

Additions in **this release** (2025-05-14)
──────────────────────────────────────────
● EOS-aware early exit  stop computing for sequences that already ended.
● GPU-memory guard  halves `beam_width` when memory > 90 %.
● Loss-convergence early-stop (per time-step) with `convergence_tol_pct`
  and `convergence_patience`.
● **Global** early-stop: after the first convergence event, _all later
  time-steps skip surrogate training entirely_ (decode only).
"""

from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from src.beamclean.scorer import JointProbabilityScoring
from src.beamclean.surrogate import (
    IsotropicGaussianSurrogate,
    IsotropicGaussianSurrogateWithMean,
    IsotropicL1LaplacianSurrogate,
    IsotropicL2LaplacianSurrogate,
)
from src.beamclean.utils import log_memory_usage
from src.models import get_prior_model

if TYPE_CHECKING:
    import transformers
    from jaxtyping import Float, Int
    from omegaconf import DictConfig
    from torch.utils.data import DataLoader

    from src.beamclean.surrogate import SurrogateModel

# Memory threshold for beam width reduction (90%)
BOS_TOKEN_ID = 128_000  # token-id placeholder (unused here but kept for ref)
EOS_TOKEN_ID = 0


class BeamCleanTrainer:
    """BeamClean: Beam-search decoding assisted by a surrogate likelihood."""

    # ─────────────────────────────────────────────────────────────────── #
    # INITIALISATION
    # ─────────────────────────────────────────────────────────────────── #
    def __init__(
        self,
        config: DictConfig,
        device: torch.device,
        embedding_table: Tensor,
        vocab_token_ids: Tensor,
        bos_token_id: int,
    ) -> None:
        self.config = config
        self.device = device

        # beam search parameters ----------------------------------------
        self.beam_width = config.beam_width
        self.vocab_token_ids = vocab_token_ids
        self.vocab_size = vocab_token_ids.numel()
        self.bos_token_id = bos_token_id
        # EOS & early-stop state ----------------------------------------
        self.finished: torch.BoolTensor | None = None
        self.stop_training = False  # ⬅ global flag
        self.tol_pct = getattr(config, "convergence_tol_pct", 0.5) / 100.0
        self.patience = getattr(config, "convergence_patience", 5)

        # models ---------------------------------------------------------
        self.surrogate_model: SurrogateModel = self._make_surrogate(device)
        self.scorer = self._make_scorer(embedding_table, device)
        self.optimizer = torch.optim.Adam(
            self.surrogate_model.parameters(),
            lr=config.learning_rate,
        )

    # ─────────────────────────────────────────────────────────────────── #
    # PRIVATE HELPERS
    # ─────────────────────────────────────────────────────────────────── #
    def _make_surrogate(self, device: torch.device) -> SurrogateModel:
        name = self.config.surrogate_model.lower()
        val = self.config.initial_param_val
        if name == "gaussian":
            return IsotropicGaussianSurrogate(initial_std=val).to(device)
        if name == "l1_laplace":
            return IsotropicL1LaplacianSurrogate(initial_scale=val).to(device)
        if name == "l2_laplace":
            return IsotropicL2LaplacianSurrogate(initial_scale=val).to(device)
        if name == "gaussian_with_mean":
            return IsotropicGaussianSurrogateWithMean(initial_std=val).to(device)
        error_msg = f"Unsupported surrogate model: {name}"
        raise ValueError(error_msg)

    def _make_scorer(
        self, embedding_table: Tensor, device: torch.device
    ) -> JointProbabilityScoring:
        """Bind surrogate & prior log-p functions into JointProbabilityScoring."""
        surrogate_logp_fn = partial(
            static_surrogate_logp_fn,  # defined in src.beamclean
            embedding_table=embedding_table,
            surrogate_model=self.surrogate_model,
        )
        prior_logp_fn = partial(
            prior_score,  # defined in src.beamclean
            vocab_token_ids=self.vocab_token_ids,
            device=device,
            pad_token_id=0,
            prior_model=get_prior_model(self.config.prior_model).to(device),
            bos_token_id=self.bos_token_id,
        )
        return JointProbabilityScoring(
            surrogate_logp_fn=surrogate_logp_fn,
            prior_logp_fn=prior_logp_fn,
            embedding_table=embedding_table,
            device=device,
        )

    # ─────────────────────────────────────────────────────────────────── #
    # MINI‑BATCH → SURROGATE SCORES
    # ─────────────────────────────────────────────────────────────────── #
    def _process_batch(
        self, batch: dict[str, Tensor], beam_ids: Tensor, t: int
    ) -> tuple[Tensor, Tensor]:
        idx = batch["sample_ids"]  # [B]
        noisy = batch["noisy_embeddings"][:, t].to(self.device)

        with torch.amp.autocast(self.device.type):
            scores = self.scorer.compute_scores(
                noisy_embeddings=noisy,
                beam_ids=beam_ids[idx],
                vocab_token_ids=self.vocab_token_ids,
                vocab_chunk_size=self.config.vocab_chunk_size,
            )  # [B, K, V]

        return idx, scores

    # ─────────────────────────────────────────────────────────────────── #
    # SURROGATE TRAINING  (returns bool ⇢ converged?)
    # ─────────────────────────────────────────────────────────────────── #
    def fit_surrogate_model(
        self,
        dataloader: DataLoader,
        num_epochs: int,
        beam_ids: Tensor,
        t: int,
    ) -> bool:
        self.surrogate_model.train()
        accumulate = getattr(self.config, "accumulate_gradients", False)

        prev_loss: float | None = None
        stable_epochs = 0

        for epoch in range(num_epochs):
            epoch_loss, n_batches = 0.0, 0
            desc = f"train e{epoch + 1}/{num_epochs}"

            if accumulate:
                self.optimizer.zero_grad()
                all_scores = []

                for batch in tqdm(dataloader, desc=desc + " (acc)", leave=False, ncols=80):
                    idx_act, batch_act = self._active_subbatch(batch, t)
                    if idx_act is None:  # no usable row in this mini-batch
                        continue
                    _, scores = self._process_batch(batch_act, beam_ids, t)
                    if self.finished is not None:
                        active = ~self.finished[idx_act]
                        if not active.any():
                            continue
                        scores = scores[active]
                    all_scores.append(scores)

                if all_scores:  # might be empty
                    scores = torch.cat(all_scores).view(-1, self.vocab_size)
                    loss = -torch.mean(torch.logsumexp(scores, 1))
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss = loss.item()

            else:  # step-per-batch
                for batch in tqdm(dataloader, desc=desc + " (step)", leave=False, ncols=80):
                    idx_act, batch_act = self._active_subbatch(batch, t)
                    if idx_act is None:  # no usable row in this mini-batch
                        continue
                    _, scores = self._process_batch(batch_act, beam_ids, t)
                    if self.finished is not None:
                        active = ~self.finished[idx_act]
                        if not active.any():
                            continue
                        scores = scores[active]

                    loss = -torch.mean(torch.logsumexp(scores.view(-1, self.vocab_size), 1))
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                if n_batches:
                    epoch_loss /= n_batches

            if epoch_loss:  # 0 if no active data
                logging.info("t=%d epoch=%d loss=%.4f", t, epoch + 1, epoch_loss)

            # convergence check ------------------------------------------
            if prev_loss is not None:
                rel = abs(epoch_loss - prev_loss) / max(abs(prev_loss), 1e-12)
                stable_epochs = stable_epochs + 1 if rel < self.tol_pct else 0
            prev_loss = epoch_loss

            if stable_epochs >= self.patience:
                logging.info(
                    "Loss converged (<±%.2f%%) for %d epochs → stop training",
                    self.tol_pct * 100,
                    self.patience,
                )
                self.surrogate_model.log()
                torch.cuda.empty_cache()
                return True  # ← converged

            self.surrogate_model.log()
            torch.cuda.empty_cache()

        return False  # ← not converged yet

    # ────────────────────────────────────────────────────────────────
    #  DECODE ONE POSITION  (skips finished samples, robust to length)
    # ────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def decode_step(
        self,
        dataloader: DataLoader,
        beam_log_p: Tensor,
        beam_ids: Tensor,
        t: int,
    ) -> tuple[Tensor, Tensor]:
        self.surrogate_model.eval()

        updates: list[tuple[Tensor, Tensor]] = []
        N = beam_ids.size(0)

        for batch in tqdm(dataloader, desc="decode", leave=False, ncols=80):
            # keep only rows that are “alive” and long enough for step t
            idx_act, batch_act = self._active_subbatch(batch, t)

            if idx_act is None:  # nothing useful in this mini-batch
                continue

            _, scores = self._process_batch(batch_act, beam_ids, t)

            # idx_act is already filtered for self.finished inside _active_subbatch
            idx = idx_act
            B = idx.size(0)

            scores = scores.cpu()
            if t == 0:  # K_prev == 1 at t = 0
                scores = scores[:, 0, :].unsqueeze(1)
                total = scores
            else:
                total = beam_log_p[idx].unsqueeze(-1) + scores

            flat_total = total.view(B, -1)  # [B, K_prev*V]
            best, flat = torch.topk(flat_total, self.beam_width, dim=1)

            parent = flat // self.vocab_size  # [B, K]
            tok_id = flat % self.vocab_size  # [B, K]
            new_tok = self.vocab_token_ids[tok_id]  # [B, K]

            prefix = beam_ids[idx].gather(1, parent.unsqueeze(-1).expand(-1, -1, t))
            updates.append((idx, torch.cat([prefix, new_tok.unsqueeze(-1)], dim=-1)))
            beam_log_p[idx] = best

        # ------------------------------------------------------------------
        # Build the new beam tensor:
        #   • start from previous prefixes
        #   • add an EOS column (so untouched rows stay valid)
        #   • scatter updates for active rows
        # ------------------------------------------------------------------
        eos_col = beam_ids.new_full((N, self.beam_width, 1), EOS_TOKEN_ID)
        new_beam = torch.cat([beam_ids, eos_col], dim=-1).clone()

        for idx, slice_ in updates:
            new_beam[idx] = slice_

        return new_beam, beam_log_p

    # ─────────────────────────────────────────────────────────────────── #
    # MAIN LOOP
    # ─────────────────────────────────────────────────────────────────── #
    def run(self, dataloader: DataLoader) -> tuple[Tensor, Tensor]:
        N = len(dataloader.dataset)
        T_max = dataloader.dataset.max_seq_len
        beam_ids = torch.empty(N, self.beam_width, 0, dtype=torch.long)
        beam_log_p = torch.zeros(N, self.beam_width)

        self.finished = torch.zeros(N, dtype=torch.bool)

        for t in tqdm(range(T_max), desc="t-step"):
            if self.finished.all():
                break

            # -- surrogate training (skip if already converged globally) --
            if not self.stop_training:
                self.stop_training = self.fit_surrogate_model(
                    dataloader, self.config.num_epochs, beam_ids, t
                )

            # -- decoding step --
            beam_ids, beam_log_p = self.decode_step(dataloader, beam_log_p, beam_ids, t)

            # mark finished sequences
            self.finished |= beam_ids[:, 0, -1] == EOS_TOKEN_ID
            beam_log_p[self.finished] = float("-inf")

            self._memory_guard(beam_ids, beam_log_p)

        return beam_ids[:, 0], beam_log_p[:, 0]

    # ─────────────────────────────────────────────────────────────────── #
    # GPU MEMORY GUARD
    # ─────────────────────────────────────────────────────────────────── #
    def _memory_guard(self, beam_ids: Tensor, beam_log_p: Tensor) -> None:
        pct = log_memory_usage()
        if pct > self.config.memory_threshold and self.beam_width > 1:
            new_bw = max(1, self.beam_width // 2)
            logging.info(
                "GPU at %.1f %% — shrinking beam %d → %d",
                pct * 100,
                self.beam_width,
                new_bw,
            )
            self.beam_width = new_bw
            beam_ids.resize_(beam_ids.size(0), new_bw, beam_ids.size(2))
            beam_log_p.resize_(beam_log_p.size(0), new_bw)

    def _active_subbatch(
        self,
        batch: dict[str, Tensor],
        t: int,
    ) -> tuple[torch.Tensor, dict[str, Tensor]] | tuple[None, None]:
        """Returns active rows from batch.

        Returns:
        -------
        idx_act : 1-D LongTensor         sample_ids of active rows
        batch_act : dict[str, Tensor]    same as input but first-dim sliced

        If *no* row is active, returns (None, None).
        """
        # sample_ids → tensor
        idx_full = torch.as_tensor(batch["sample_ids"], dtype=torch.long)

        # 1) still unfinished?
        if self.finished is None:
            active = torch.ones_like(idx_full, dtype=torch.bool)
        else:
            active = ~self.finished[idx_full]

        # 2) long enough for this t?
        local_T = batch["noisy_embeddings"].size(1)
        active &= t < local_T

        if not active.any():
            return None, None

        # slice every tensor in the batch dict
        batch_act = {k: v[active] if torch.is_tensor(v) else v for k, v in batch.items()}
        batch_act["sample_ids"] = idx_full[active].tolist()
        return idx_full[active], batch_act


class BeamCleanTrainDecodeOnly(BeamCleanTrainer):
    """BeamClean variant that **never re-trains** the surrogate; it only decodes."""

    # ------------------------------------------------------------------ #
    # INIT (same as parent, nothing extra)
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        config: DictConfig,
        device: torch.device,
        embedding_table: Tensor,
        vocab_token_ids: Tensor,
        bos_token_id: int,
    ) -> None:
        super().__init__(config, device, embedding_table, vocab_token_ids, bos_token_id)

    # ------------------------------------------------------------------ #
    # RUN (decode-only, with EOS skipping & memory guard)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def run(self, dataloader: DataLoader) -> tuple[Tensor, Tensor]:
        """Decode the full sequence length **without further surrogate training**.

        Returns:
        -------
        decoded_ids : Tensor [N,T_max]
            Final token-ID sequence for each sample (uses best beam = 0).
        log_p       : Tensor [N]
            Corresponding log-probabilities.
        """
        N = len(dataloader.dataset)
        T_max = dataloader.dataset.max_seq_len
        beam_ids = torch.empty(N, self.beam_width, 0, dtype=torch.long)
        beam_log_p = torch.zeros(N, self.beam_width)

        # every sample unfinished at t = 0
        self.finished = torch.zeros(N, dtype=torch.bool)

        for t in tqdm(range(T_max), desc="decode-only t-step"):
            if self.finished.all():
                break

            # ------------ single decoding step (parent method handles skipping) --
            beam_ids, beam_log_p = self.decode_step(
                dataloader=dataloader,
                beam_log_p=beam_log_p,
                beam_ids=beam_ids,
                t=t,
            )

            # ------------ mark sequences that just emitted EOS ---------------
            self.finished |= beam_ids[:, 0, -1] == EOS_TOKEN_ID
            beam_log_p[self.finished] = float("-inf")  # freeze their score

            # ------------ optional: shrink beam if GPU memory is tight -------
            self._memory_guard(beam_ids, beam_log_p)
            torch.cuda.empty_cache()

        # return best beam (index 0) for every sample
        return beam_ids[:, 0], beam_log_p[:, 0]


# Trainer class mapping based on configuration
TRAINER_CLASSES = {
    True: BeamCleanTrainDecodeOnly,
    False: BeamCleanTrainer,
}


def static_surrogate_logp_fn(
    noisy_embeddings: Tensor,
    cand_token_ids: Tensor,
    surrogate_model: SurrogateModel,
    embedding_table: Tensor,
) -> Tensor:
    """Static surrogate log probability function."""
    noise_stats = surrogate_model(None)
    candidate_embeddings = embedding_table[cand_token_ids]
    return surrogate_model.log_pdf(noisy_embeddings, candidate_embeddings, noise_stats)


def prior_lm_fn(
    _x: torch.Tensor,
    beam_tokens: torch.Tensor,
) -> torch.Tensor:
    """Mock function to compute prior likelihood scores."""
    return torch.sum(beam_tokens, dim=-1).cuda()  # Example operation


def prior_score(
    input_token_ids: Float[Tensor, "batch beam time"],
    vocab_token_ids: torch.Tensor,
    prior_model: torch.nn.Module,
    device: torch.device,
    pad_token_id: int,
    bos_token_id: int,
) -> torch.Tensor:
    """Compute log probabilities of candidate tokens using a prior language model.

    Args:
        input_token_ids: Input token IDs [batch_size, beam_width, chunk_size, time_step]
        vocab_token_ids: Vocabulary token IDs
        prior_model: Language model to use for scoring
        device: Device to run computations on
        pad_token_id: ID of padding token
        bos_token_id: ID of BOS token

    Returns:
        Log probabilities of candidate tokens [batch_size, beam_width, chunk_size]
    """
    # Add BOS token to the input token ids
    batch_size, beam_width, _time_step = input_token_ids.shape
    input_token_ids = torch.cat(
        [
            torch.ones((*input_token_ids.shape[:2], 1)).long() * bos_token_id,
            input_token_ids,
        ],
        dim=-1,
    )
    input_token_ids: Float[Tensor, "batch beam time + 1"] = input_token_ids.to(device)
    vocab_token_ids = vocab_token_ids.to(device)
    flat_token_ids = input_token_ids.view(-1, input_token_ids.size(-1))
    attention_mask = flat_token_ids != pad_token_id
    attention_mask = attention_mask.to(device)
    with torch.no_grad():
        outputs = prior_model(input_ids=flat_token_ids, attention_mask=attention_mask)

    logits = outputs.logits
    # Get the last token's logits
    last_token_logits = logits[:, -1, :][:, vocab_token_ids]
    # Apply softmax to get probabilities
    return torch.nn.functional.log_softmax(last_token_logits, dim=-1).view(
        batch_size, beam_width, -1
    )


def beam_clean_decode(
    embedding_table: Float[Tensor, "V d"],
    vocab_token_ids: Tensor | np.ndarray,
    dataloader: DataLoader,
    prior_model: transformers.AutoModelForCausalLM,
    surrogate_model: SurrogateModel,
    beam_width: int,
    pad_token_id: int,
    bos_token_id: int,
    max_seq_len: int,
    *,
    vocab_chunk_size: int = 2_048,
) -> dict[int, np.ndarray]:
    """BeamClean decoding with memory-friendly chunked surrogate scoring.

    This implementation mirrors the *chunked* processing strategy used in
    :pyfunc:`compute_scores` to avoid materialising large `(B, Z, V)` tensors
    inside the surrogate while still recovering the exact same numerical result.

    Args:
        embedding_table: Token embedding matrix of shape ``(V, d)``.
        vocab_token_ids: Vocabulary token IDs (``(V,)``) aligned with
            ``embedding_table``. Accepts either a **torch** or **numpy** array.
        dataloader: Yields dictionaries with keys ``"sample_ids"``,
            ``"seq_lens"`` and ``"noisy_embeddings"``.
        prior_model: Causal-LM that provides the prior log-probabilities.
        surrogate_model: Instance that exposes a ``log_pdf`` method and,
            optionally, a ``noise_stats`` predictor.
        beam_width: Number of beams ``Z``.
        pad_token_id: PAD token ID for ``prior_score``.
        bos_token_id: BOS token ID placed at sequence start.
        max_seq_len: Maximum sequence length ``T`` to decode.
        vocab_chunk_size: Process at most this many vocabulary items per call
            to ``surrogate_model`` (default ``2048``).

    Returns:
        Mapping from *sample ID* to the **decoded** token ID sequence as a
        NumPy array.
    """
    # ---------------------------------------------------------------------
    # 0. House-keeping & type harmonisation
    # ---------------------------------------------------------------------
    if isinstance(vocab_token_ids, np.ndarray):
        # Move to torch on the same device as the embedding table for fast indexing.
        vocab_token_ids = torch.as_tensor(vocab_token_ids, device=embedding_table.device)
    else:
        vocab_token_ids = vocab_token_ids.to(embedding_table.device)

    V: int = embedding_table.size(0)  # vocabulary size
    device: torch.device = embedding_table.device

    best_decoded_sequences: dict[int, np.ndarray] = {}
    sequence_lens: dict[int, int] = {}

    # ---------------------------------------------------------------------
    # 1. Iterate over dataset batches
    # ---------------------------------------------------------------------
    for batch in tqdm(dataloader, desc="Batches", position=0):
        # -----------------------------------------------------------------
        # 1-A. Unpack batch & bookkeeping
        # -----------------------------------------------------------------
        sample_ids: list[int] = batch["sample_ids"]  # length B
        seq_lens_batch: list[int] = batch["seq_lens"]  # length B
        sequence_lens.update({sid: seq_lens_batch[i] for i, sid in enumerate(sample_ids)})

        B: int = len(sample_ids)  # batch size
        noisy_embeddings: Float[Tensor, "B T d"] = batch["noisy_embeddings"].to(device)

        # -----------------------------------------------------------------
        # 1-B. Initialise beam containers
        # -----------------------------------------------------------------
        # batch_beam_sequences :: (B, Z, t) — start with BOS, t = 1, Z = 1
        batch_beam_sequences: Int[np.ndarray, "B 1 1"] = np.full(
            (B, 1, 1), bos_token_id, dtype=np.int32
        )
        # batch_beam_scores :: (B, Z, 1)
        batch_beam_scores: Float[np.ndarray, "B 1 1"] = np.zeros((B, 1, 1), dtype=np.float32)

        # -----------------------------------------------------------------
        # 2. Time-step loop (t = 0..T-1)
        # -----------------------------------------------------------------
        for time_idx in tqdm(range(max_seq_len), desc="Timesteps", leave=False, position=1):
            # real_beam_width == 1 for the first step (BOS) then == beam_width
            Z: int = 1 if time_idx == 0 else beam_width

            # current_noisy_embeddings :: (B, d)
            current_noisy_embeddings: Float[Tensor, "B d"] = noisy_embeddings[:, time_idx, :]

            # -----------------------------------------------------------------
            # 2-A. Compute surrogate log-likelihood *chunked* over V
            # -----------------------------------------------------------------
            noise_stats = surrogate_model(None)  # model-specific — can be None
            logp_surr_chunks: list[np.ndarray] = []  # each chunk -> (B, 1, C)

            for start in range(0, V, vocab_chunk_size):
                end: int = min(start + vocab_chunk_size, V)
                cand_embeddings: Float[Tensor, "C d"] = embedding_table[start:end]  # (C, d)

                # logp_chunk :: (B, C)
                logp_chunk: Float[Tensor, "B C"] = surrogate_model.log_pdf(
                    noisy_embeddings=current_noisy_embeddings,
                    candidate_embeddings=cand_embeddings,
                    noise_stats=noise_stats,
                )
                logp_surr_chunks.append(logp_chunk.detach().cpu().numpy()[:, None, :])

            # dist_log_likelihoods :: (B, 1, V)
            dist_log_likelihoods: Float[np.ndarray, "B 1 V"] = np.concatenate(
                logp_surr_chunks, axis=2
            )

            # -----------------------------------------------------------------
            # 2-B. Compute prior log-probabilities over V (no chunking)
            # -----------------------------------------------------------------
            # input_token_ids :: (B, Z, t)
            input_token_ids = torch.tensor(batch_beam_sequences, device=device)

            # batch_prior_score :: (B, 1, V)
            batch_prior_score: Float[np.ndarray, "B 1 V"] = (
                prior_score(
                    input_token_ids=input_token_ids,
                    vocab_token_ids=vocab_token_ids,
                    prior_model=prior_model,
                    device=device,
                    pad_token_id=pad_token_id,
                    bos_token_id=bos_token_id,
                )
                .cpu()
                .numpy()
            )

            # -----------------------------------------------------------------
            # 2-C. Merge scores and select top-Z extensions
            # -----------------------------------------------------------------
            # new_scores :: (B, 1, V) — broadcasting (B,Z,1)+(B,1,V)
            new_scores: Float[np.ndarray, "B 1 V"] = (
                dist_log_likelihoods + batch_beam_scores + batch_prior_score
            )
            # Flatten beam dimension → (B, V)
            new_scores_flat: Float[np.ndarray, "B V"] = new_scores.reshape(B, -1)

            # chosen_indices :: (B, Z)
            chosen_indices: Int[np.ndarray, "B Z"] = np.argsort(new_scores_flat, axis=1)[:, -Z:]

            # Map flat indices → (beam_idx, token_idx)
            chosen_beams: Int[np.ndarray, "B Z t"] = batch_beam_sequences[
                np.arange(B)[:, None],
                chosen_indices // V,  # integer division picks the beam index
            ]
            chosen_tokens: Int[np.ndarray, "B Z 1"] = (
                vocab_token_ids[(chosen_indices % V)[:, :, None]].cpu().numpy()
            )

            # Extend sequences and update scores
            batch_beam_sequences = np.concatenate(
                (chosen_beams, chosen_tokens), axis=-1
            )  # (B,Z,t+1)
            batch_beam_scores = new_scores_flat[np.arange(B)[:, None], chosen_indices].reshape(
                B, Z, 1
            )  # (B,Z,1)

        # -----------------------------------------------------------------
        # 3. Finalise: keep best beam (last one after argsort) per sample
        # -----------------------------------------------------------------
        best_beams_batch: Int[np.ndarray, "B t"] = batch_beam_sequences[:, -1, 1:]  # drop BOS
        for i, sid in enumerate(sample_ids):
            best_decoded_sequences[sid] = best_beams_batch[i, : sequence_lens[sid]]

    return best_decoded_sequences
