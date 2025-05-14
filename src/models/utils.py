from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertModel,
    GPT2Model,
    GPT2Tokenizer,
    OPTForSequenceClassification,
    T5Model,
)

from src.models.surrogate_model import (
    SurrogateModelConstant,
)


class SurrogateModelType(Enum):
    CONSTANT = "constant"
    CONSTANT_AXIAL = "constant_axial"


if TYPE_CHECKING:
    import omegaconf


BASE_MODEL_CHOICES = [
    "stevhliu/my_awesome_model",
    "gpt2",
    "gpt2-xl",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    "facebook/opt-350m",
    "facebook/opt-125m",
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-7b-chat-hf",
    "bert-base-uncased",
    "bert-large-uncased",
    "t5-small",
    "t5-base",
    "t5-large",
    "gpt2-medium",
    "gpt2-large",
]

BASE_PRECISION_CHOICES = [torch.float32]


def get_prior_model(model_path: str) -> AutoModelForCausalLM:
    """Load the prior model from the specified path."""
    return AutoModelForCausalLM.from_pretrained(model_path)


def get_surrogate_model(config: omegaconf.DictConfig) -> torch.nn.Module:
    """Get the mimic model based on the configuration."""
    surrogate_model_type = SurrogateModelType(config.attack.surrogate.type)

    if surrogate_model_type == SurrogateModelType.CONSTANT:
        return SurrogateModelConstant(
            mean=config.privacy.const_noise_params.gauss.initial_mean,
            std=config.privacy.const_noise_params.gauss.initial_std,
        )

    if surrogate_model_type == SurrogateModelType.CONSTANT_AXIAL:
        # Make mean and variance np arrays with the size of the input dim
        mean = np.repeat(
            config.privacy.const_noise_params.initial_mean,
            config.data.embedding_dim,
        )
        variance = np.repeat(
            config.privacy.const_noise_params.initial_variance,
            config.data.embedding_dim,
        )
        return SurrogateModelConstant(
            mean=mean,
            std=variance,
        )

    msg = f"Unknown noise function: {surrogate_model_type}"
    raise ValueError(msg)


def get_target_model(config: omegaconf.DictConfig) -> tuple[AutoTokenizer, torch.nn.Module]:
    """Get target model and tokenizer based on configuration.

    Args:
        config: Configuration object containing model settings

    Returns:
        Tuple containing:
            - Tokenizer for the model
            - Base model instance
    """
    if config.privacy.embedding_model_path == "stevhliu/my_awesome_model":
        base_model = AutoModelForSequenceClassification.from_pretrained(
            config.privacy.embedding_model_path
        )
        tokenizer = AutoTokenizer.from_pretrained(config.privacy.embedding_model_path)
    elif config.privacy.embedding_model_path in (
        "bert-base-uncased",
        "bert-large-uncased",
    ):
        base_model = BertModel.from_pretrained(config.privacy.embedding_model_path)
        tokenizer = AutoTokenizer.from_pretrained(config.privacy.embedding_model_path)
    elif config.privacy.embedding_model_path in (
        "THUDM/chatglm2-6b-int4",
        "THUDM/chatglm2-6b",
    ):
        base_model = AutoModel.from_pretrained(
            config.privacy.embedding_model_path, trust_remote_code=True
        ).half()  # FP16 by default
        tokenizer = AutoTokenizer.from_pretrained(
            config.privacy.embedding_model_path, trust_remote_code=True
        )
    elif "gpt2" in config.privacy.embedding_model_path:
        tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(
            config.privacy.embedding_model_path
        )
        tokenizer.pad_token = tokenizer.eos_token
        base_model = GPT2Model.from_pretrained(config.privacy.embedding_model_path)
    elif "opt" in config.privacy.embedding_model_path:
        tokenizer = AutoTokenizer.from_pretrained(config.privacy.embedding_model_path)
        base_model = OPTForSequenceClassification.from_pretrained(
            config.privacy.embedding_model_path
        )
    elif "llama" in config.privacy.embedding_model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(config.privacy.embedding_model_path)
        tokenizer.pad_token = tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            config.privacy.embedding_model_path  # , torch_dtype=config.base_precision
        )
        # if config.llama_dir is not None:
        #     base_model = AutoModelForCausalLM.from_pretrained(
        #         config.llama_dir, torch_dtype=config.base_precision
        #     )
        # else:
        #     base_model = AutoModelForCausalLM.from_pretrained(
        #         config.privacy.embedding_model_path  # , torch_dtype=config.base_precision
        #     )
    elif "t5" in config.privacy.embedding_model_path:
        tokenizer = AutoTokenizer.from_pretrained(config.privacy.embedding_model_path)
        base_model = T5Model.from_pretrained(config.privacy.embedding_model_path)
    else:
        raise ValueError(f"Unknown base model: {config.privacy.embedding_model_path}")

    return tokenizer, base_model
