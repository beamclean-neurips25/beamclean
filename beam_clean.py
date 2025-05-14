"""BeamClean main entry point — optional stages & baseline PII.

================================================================
A flexible runner for BeamClean experiments that lets you selectively enable
or disable decoding stages **at runtime** via Hydra config flags:

```yaml
run:
  beam_clean: true         # run BeamClean stage
  nearest_neighbor: false  # skip NN baseline
privacy:
  evaluate_pii: true       # requires OPENAI_API_KEY and dataset metadata
```

Key features
------------
1. Pluggable stages — `beam_clean`, `nearest_neighbor`, and always-present `original` baseline.
2. Unified artefact file — `results.json` contains config, metrics,
and all sequences keyed by stage.
3. Optional PII evaluation — gracefully skipped if requirements not met.
4. Strict type hints & runtime shape checks via jaxtyping
(can be disabled by `JAXTYPING_DISABLE_RUNTIME=1`).
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import dotenv
import dspy
import hydra
import numpy as np
import omegaconf
import torch
from jaxtyping import install_import_hook
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.beamclean.trainer import TRAINER_CLASSES
from src.beamclean.utils import setup_data_and_models, setup_device, setup_environment
from src.nearest_neighbor import nearest_neighbor_decode
from src.utils.metrics import LLMJudge, calculate_decoding_accuracy

if TYPE_CHECKING:  # pragma: no cover
    from transformers import PreTrainedModel, PreTrainedTokenizer

    from src.dataset.base_dataset import BaseDataset

# ---------------------------------------------------------------------------
# Global constants & logging
# ---------------------------------------------------------------------------

PAD_TOKEN_ID = 0
dotenv.load_dotenv(override=True)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
install_import_hook("src.beamclean", ("jaxtyping.runtime_checks",))

_LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Dataclasses for serialisation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SequenceRecord:
    """Stores original ↔ decoded sequence pair for a single sample."""

    id: int
    original_text: str
    decoded_text: str
    original_ids: list[int]
    decoded_ids: list[int]


@dataclass(slots=True)
class RunArtefacts:
    """Top-level container persisted as JSON."""

    config: dict[str, Any]
    metrics: dict[str, dict[str, float | None]]
    sequences: dict[str, list[SequenceRecord]] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """Convert to JSON."""
        return {
            "config": self.config,
            "metrics": self.metrics,
            "sequences": {
                stage: [asdict(r) for r in recs] for stage, recs in self.sequences.items()
            },
        }

    @staticmethod
    def from_json(json_data: dict[str, Any]) -> RunArtefacts:
        """Convert from JSON."""
        config = json_data["config"]
        metrics = json_data["metrics"]
        sequences = {
            stage: [SequenceRecord(**r) for r in recs]
            for stage, recs in json_data["sequences"].items()
        }
        return RunArtefacts(config=config, metrics=metrics, sequences=sequences)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _dataset_supports_pii(dataset: BaseDataset) -> bool:
    """Return True if every sample exposes a `metadata['pii_str']` field."""
    try:
        return all(bool(getattr(s, "metadata", {}).get("pii_str", "")) for s in dataset)
    except Exception:  # noqa: BLE001 pylint: disable=broad-exception-caught
        return False


# ---- LLM judge ------------------------------------------------------------


def _init_llm_judge() -> LLMJudge:
    """Instantiate DSPy LLMJudge with local SGLang + OpenAI backend."""
    api_key = os.getenv("OPENAI_API_KEY")
    openai_lm = dspy.LM(model="openai/gpt-4o", max_tokens=4000, api_key=api_key)
    dspy.configure(experimental=True)

    judge = LLMJudge()
    judge.set_lm(openai_lm)
    return judge


# ---- Sequence conversion helpers -----------------------------------------
def _records_from_ids(
    dataset: BaseDataset, tokenizer: PreTrainedTokenizer, decoded_ids: torch.Tensor
) -> list[SequenceRecord]:
    """Map model outputs back to texts & ids."""
    recs: list[SequenceRecord] = []
    for sample in dataset:
        sid = sample.sample_id
        orig_ids_t = torch.from_numpy(sample.input_token_ids)
        dec_ids_t = decoded_ids[sid, : len(orig_ids_t)]
        recs.append(
            SequenceRecord(
                id=sid,
                original_text=tokenizer.decode(orig_ids_t),
                decoded_text=tokenizer.decode(dec_ids_t),
                original_ids=orig_ids_t.tolist(),
                decoded_ids=dec_ids_t.tolist(),
            )
        )
    return recs


def _original_records(dataset: BaseDataset, tokenizer: PreTrainedTokenizer) -> list[SequenceRecord]:
    """Reference records where decoded == original (upper-bound)."""
    recs: list[SequenceRecord] = []
    for sample in dataset:
        ids = torch.from_numpy(sample.input_token_ids)
        recs.append(
            SequenceRecord(
                id=sample.sample_id,
                original_text=tokenizer.decode(ids),
                decoded_text=tokenizer.decode(ids),
                original_ids=ids.tolist(),
                decoded_ids=ids.tolist(),
            )
        )
    return recs


# ---- Metric helpers -------------------------------------------------------
def _accuracy(dataset: BaseDataset, recs: list[SequenceRecord]) -> float:
    # Use torch's pad_sequence with batch_first=True
    ids = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(r.decoded_ids) for r in recs], batch_first=True, padding_value=PAD_TOKEN_ID
    )
    return float(calculate_decoding_accuracy(dataset=dataset, decoded_ids=ids, pad_id=PAD_TOKEN_ID))


def _pii_leakage(dataset: BaseDataset, recs: list[SequenceRecord], judge: LLMJudge) -> float:
    scores: list[float] = []
    for sample, rec in zip(dataset, recs, strict=True):
        pii_raw = sample.metadata.get("pii_str", "")  # type: ignore[attr-defined]
        pii_list = [p for p in pii_raw.split("||") if p]
        try:
            leaked = judge.fact_checker(pii=pii_list, prompt=rec.decoded_text).num_pii_leaked
            scores.append(leaked / len(pii_list) if pii_list else 0.0)
        except Exception as exc:  # noqa: BLE001 pylint: disable=broad-exception-caught
            _LOG.warning("PII scoring error: %s", exc)
            scores.append(float("nan"))
    return float(sum(scores) / len(scores)) if scores else float("nan")


# ---- Dataloader / Vocab helpers ------------------------------------------


def _prepare_vocab(
    cfg: DictConfig, target_model: PreTrainedModel
) -> tuple[torch.Tensor, torch.Tensor]:
    table: torch.Tensor = target_model.get_input_embeddings().weight  # [V, d]
    ids_path: str | None = cfg.get("beam_clean", {}).get("candidate_token_ids_path")
    if ids_path is None:
        return torch.arange(table.size(0)), table
    return torch.from_numpy(np.load(pathlib.Path(ids_path))), table


def _build_beam_clean_loader(cfg: DictConfig, dataset: BaseDataset) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )


def _build_nearest_neighbor_loader(cfg: DictConfig, dataset: BaseDataset) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )


# ---- Config validation ----------------------------------------------------


def _validate_config(cfg: DictConfig) -> None:
    required = [
        "data.truncated_seq_len",
        "data.batch_size",
        "run_dir",
        # BeamClean params (validate even if beam off; safer)
        "beam_clean.beam_width",
        "beam_clean.num_epochs",
        "beam_clean.learning_rate",
    ]
    for _field in required:
        if omegaconf.OmegaConf.select(cfg, _field) is None:
            error_msg = f"Missing required config field: {_field}"
            _LOG.error(error_msg)
            raise ValueError(error_msg)
    if cfg.beam_clean.beam_width <= 0:
        error_msg = "beam_clean.beam_width must be >0"
        _LOG.error(error_msg)
        raise ValueError(error_msg)
    if cfg.beam_clean.num_epochs <= 0:
        error_msg = "beam_clean.num_epochs must be >0"
        _LOG.error(error_msg)
        raise ValueError(error_msg)
    if cfg.beam_clean.learning_rate <= 0:
        error_msg = "beam_clean.learning_rate must be >0"
        _LOG.error(error_msg)
        raise ValueError(error_msg)


# ---------------------------------------------------------------------------
# Main entry via Hydra
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="config", config_name="main")
def main(cfg: DictConfig) -> None:
    """Main entry point."""
    _validate_config(cfg)
    setup_environment(config=cfg)
    device = setup_device()

    # Stage toggles
    beam_on: bool = cfg.get("run", {}).get("beam_clean", True)
    nn_on: bool = cfg.get("run", {}).get("nearest_neighbor", True)
    pii_on: bool = bool(cfg.get("evaluate_pii", False))

    dataset, tokenizer, target_model = setup_data_and_models(config=cfg)
    vocab_ids, embed_table = _prepare_vocab(cfg, target_model)
    embed_table = embed_table.to(device)

    stages: dict[str, list[SequenceRecord]] = {"original": _original_records(dataset, tokenizer)}

    # ---------------- BeamClean ----------------
    beam_time = None
    if beam_on:
        loader = _build_beam_clean_loader(cfg, dataset)
        trainer_class = TRAINER_CLASSES[cfg.beam_clean.decode_only]
        trainer = trainer_class(
            device=device,
            vocab_token_ids=vocab_ids,
            embedding_table=embed_table,
            config=cfg.beam_clean,
            bos_token_id=tokenizer.bos_token_id,
        )
        start_time = time.time()
        dec_beam, _ = trainer.run(dataloader=loader)
        beam_time = time.time() - start_time
        beam_time = time.strftime("%H:%M:%S", time.gmtime(beam_time))
        stages["beam_clean"] = _records_from_ids(dataset, tokenizer, dec_beam)

    # -------------- Nearest Neighbor ----------
    # Make dataloader batch size 1
    nn_time = None
    if nn_on:
        loader = _build_nearest_neighbor_loader(cfg, dataset)
        start_time = time.time()
        dec_nn = nearest_neighbor_decode(
            dataloader=loader,
            embedding_table=embed_table,
            vocab_token_ids=vocab_ids,
            pad_id=PAD_TOKEN_ID,
        )[0]
        nn_time = time.time() - start_time
        nn_time = time.strftime("%H:%M:%S", time.gmtime(nn_time))
        stages["nearest_neighbor"] = _records_from_ids(dataset, tokenizer, dec_nn)

    # ---------------- Metrics -----------------
    metrics: dict[str, dict[str, float | None]] = {"accuracy": {}, "pii": {}}
    metrics["time"] = {"beam_clean": beam_time, "nearest_neighbor": nn_time}

    for name, recs in stages.items():
        if name != "original":  # original accuracy=1 implicitly
            metrics["accuracy"][name] = _accuracy(dataset, recs)
        else:
            metrics["accuracy"][name] = 100.0

    if pii_on and _dataset_supports_pii(dataset) and os.getenv("OPENAI_API_KEY"):
        judge = _init_llm_judge()
        for name, recs in stages.items():
            metrics["pii"][name] = _pii_leakage(dataset, recs, judge)
    else:
        metrics["pii"] = dict.fromkeys(stages)

    artefacts = RunArtefacts(
        config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
        metrics=metrics,
        sequences=stages,
    )

    _persist(pathlib.Path(cfg.run_dir), artefacts)
    _LOG.info("Finished. Results saved to %s/results.json", cfg.run_dir)


# ---- Persist --------------------------------------------------------------


def _persist(run_dir: pathlib.Path, artefacts: RunArtefacts) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "results.json").open("w", encoding="utf-8") as fp:
        json.dump(artefacts.to_json(), fp, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Script entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    main()  # pylint: disable=no-value-for-parameter
