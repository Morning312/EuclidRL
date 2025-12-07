from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Optional

import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig as TRLPPOConfig, PPOTrainer

from prover.configs import PPOConfig
from prover.env import LeanProofEnv


class RolloutExample:
    def __init__(self, prompt: str, theorem: Optional[str] = None) -> None:
        self.prompt = prompt
        self.theorem = theorem


class RolloutDataset(Dataset):
    def __init__(self, path: str) -> None:
        self.items: list[RolloutExample] = []
        for line in Path(path).read_text().splitlines():
            data = json.loads(line)
            self.items.append(RolloutExample(prompt=data["prompt"], theorem=data.get("theorem")))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> RolloutExample:
        return self.items[idx]


def tokenize_prompts(prompts: Iterable[str], tokenizer: PreTrainedTokenizer) -> list[torch.Tensor]:
    toks = []
    for p in prompts:
        toks.append(
            tokenizer(
                p,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=tokenizer.model_max_length,
            ).input_ids.squeeze(0)
        )
    return toks


def rollout_reward(
    response: str, example: RolloutExample, env: Optional[LeanProofEnv]
) -> float:
    if env is None or example.theorem is None:
        return float(len(response.strip()) > 0)
    state = env.reset(example.theorem)
    reward = 0.0
    for raw_tac in response.split(";"):
        tactic = raw_tac.strip()
        if not tactic:
            continue
        _, r, done, _ = env.step(tactic)
        reward += r
        if done:
            break
    return reward


def run_ppo(cfg: PPOConfig, config_path: Optional[str] = None, env: Optional[LeanProofEnv] = None) -> None:
    print("Launching PPO with config:", config_path or "defaults")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        cfg.sft_checkpoint or cfg.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    ppo_cfg = TRLPPOConfig(
        model_name=cfg.model_name,
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        ppo_epochs=cfg.ppo_epochs,
        log_with=None,
        seed=cfg.seed,
        kl_penalty=cfg.kl_penalty,
        cliprange=cfg.cliprange,
    )
    trainer = PPOTrainer(
        config=ppo_cfg,
        model=model,
        tokenizer=tokenizer,
        dataset=RolloutDataset(cfg.rollout_file),
    )

    for batch in trainer.dataloader:
        prompts: list[str] = [row.prompt for row in batch]
        tokenized_prompts = tokenize_prompts(prompts, tokenizer)
        response_tensors = trainer.generate(
            tokenized_prompts,
            max_new_tokens=cfg.max_response_length,
            pad_token_id=tokenizer.pad_token_id,
        )
        responses = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

        rewards = []
        for resp, example in zip(responses, batch):
            rewards.append(rollout_reward(resp, example, env))
        reward_tensors = [torch.tensor(r).to(trainer.accelerator.device) for r in rewards]

        stats = trainer.step(tokenized_prompts, response_tensors, reward_tensors)
        if trainer.state.global_step % cfg.log_steps == 0:
            print(f"[PPO] step={trainer.state.global_step} stats={stats}")

        if trainer.state.global_step % cfg.save_steps == 0:
            save_dir = Path(cfg.output_dir) / f"step_{trainer.state.global_step}"
            save_dir.mkdir(parents=True, exist_ok=True)
            trainer.save_pretrained(str(save_dir))

    final_dir = Path(cfg.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_pretrained(str(final_dir))
    Path(cfg.output_dir, "ppo_config_used.yaml").write_text(
        "\n".join(f"{k}: {v}" for k, v in asdict(cfg).items())
    )
