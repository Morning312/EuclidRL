from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class PPOConfig:
    model_name: str = "Qwen/Qwen2.5-Math-1.5B"
    sft_checkpoint: Optional[str] = "checkpoints/sft"
    output_dir: str = "checkpoints/ppo"
    rollout_file: str = "data/rollouts/train.jsonl"
    max_prompt_length: int = 512
    max_response_length: int = 512
    learning_rate: float = 1e-5
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    ppo_epochs: int = 4
    kl_penalty: float = 0.1
    cliprange: float = 0.2
    value_loss_coef: float = 0.2
    log_steps: int = 5
    save_steps: int = 200
    eval_steps: int = 200
    mixed_precision: Optional[str] = "bf16"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    reward_model: Optional[str] = None
    seed: int = 42


def load_ppo_config(path: Optional[str]) -> PPOConfig:
    if path is None:
        return PPOConfig()
    data = yaml.safe_load(Path(path).read_text())
    return PPOConfig(**data)
