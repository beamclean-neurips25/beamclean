"""Calculate PII leakage for each method."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import omegaconf

from beam_clean import RunArtefacts, _init_llm_judge, _pii_leakage
from src.dataset.load_dataset import get_dataset


def main() -> None:
    """Calculate PII leakage for each method."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_file", type=str, required=True)
    args = parser.parse_args()
    result_file = args.result_file
    result_file = pathlib.Path(result_file)
    with result_file.open("r", encoding="utf-8") as f:
        result = json.load(f)

    config: dict[str, Any] = result["config"]
    # Convert config to DictConfig
    config = omegaconf.DictConfig(config)
    dataset = get_dataset(config)
    run_artefacts = RunArtefacts.from_json(result)
    methods = list(run_artefacts.metrics["accuracy"].keys())
    judge = _init_llm_judge()

    for method in methods:
        pii_leakage = _pii_leakage(
            dataset=dataset, recs=run_artefacts.sequences[method], judge=judge
        )
        result["metrics"]["pii"][method] = pii_leakage

    with result_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
