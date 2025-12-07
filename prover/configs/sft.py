from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class SFTConfig:
    model_name: str = "Qwen/Qwen2.5-Math-1.5B"
    output_dir: str = "checkpoints/sft"
    train_file: str = "data/sft/train.jsonl"
    val_file: Optional[str] = "data/sft/val.jsonl"
    max_seq_length: int = 2048
    learning_rate: float = 2e-5
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 3
    warmup_steps: int = 50
    weight_decay: float = 0.0
    log_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    gradient_checkpointing: bool = True
    mixed_precision: Optional[str] = "bf16"
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
    seed: int = 42


def load_sft_config(path: Optional[str]) -> SFTConfig:
    if path is None:
        return SFTConfig()
    data = yaml.safe_load(Path(path).read_text())
    return SFTConfig(**data)
