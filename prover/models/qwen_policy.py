from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


def get_tokenizer(model_name: str) -> PreTrainedTokenizer:
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@dataclass
class PolicyInitConfig:
    model_name: str
    load_in_4bit: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


def load_qwen_policy(cfg: PolicyInitConfig) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    tokenizer = get_tokenizer(cfg.model_name)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        device_map=None,  # avoid meta/offload issues when saving
        torch_dtype=torch_dtype,
        load_in_4bit=cfg.load_in_4bit,
        trust_remote_code=True,
    ).to(device)
    if cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=list(cfg.target_modules),
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    return model, tokenizer
