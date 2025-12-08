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
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            data = json.loads(line)
            self.items.append(RolloutExample(prompt=data["prompt"], theorem=data.get("theorem")))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> RolloutExample:
        return self.items[idx]


def tokenize_prompts(
    prompts: Iterable[str], tokenizer: PreTrainedTokenizer, max_length: int = 512
) -> list[torch.Tensor]:
    toks = []
    for p in prompts:
        toks.append(
            tokenizer(
                p,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=max_length,
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
    if env is None:
        print("[WARNING] No LeanProofEnv provided. Using fallback rewards (1.0 if non-empty response).")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine torch dtype based on config and hardware
    if cfg.mixed_precision == "bf16" and torch.cuda.is_available():
        torch_dtype = torch.bfloat16
    elif cfg.mixed_precision == "fp16" and torch.cuda.is_available():
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        cfg.sft_checkpoint or cfg.model_name,
        torch_dtype=torch_dtype,
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
        vf_coef=cfg.value_loss_coef,
    )
    trainer = PPOTrainer(
        config=ppo_cfg,
        model=model,
        tokenizer=tokenizer,
        dataset=RolloutDataset(cfg.rollout_file),
    )

    step = 0
    for batch in trainer.dataloader:
        prompts: list[str] = [row.prompt for row in batch]
        tokenized_prompts = tokenize_prompts(prompts, tokenizer, cfg.max_prompt_length)

        # Generate full sequences (prompt + response)
        full_tensors = trainer.generate(
            tokenized_prompts,
            max_new_tokens=cfg.max_response_length,
            pad_token_id=tokenizer.pad_token_id,
        )

        # Extract only the response portion (remove prompt tokens)
        response_tensors = []
        for prompt_tensor, full_tensor in zip(tokenized_prompts, full_tensors):
            prompt_len = len(prompt_tensor)
            response_only = full_tensor[prompt_len:]
            response_tensors.append(response_only)

        # Decode only the response portion for reward computation
        responses = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

        rewards = []
        for resp, example in zip(responses, batch):
            rewards.append(rollout_reward(resp, example, env))
        reward_tensors = [torch.tensor(r).to(trainer.accelerator.device) for r in rewards]

        stats = trainer.step(tokenized_prompts, response_tensors, reward_tensors)
        step += 1

        if step % cfg.log_steps == 0:
            print(f"[PPO] step={step} stats={stats}")

        if step % cfg.save_steps == 0:
            save_dir = Path(cfg.output_dir) / f"step_{step}"
            save_dir.mkdir(parents=True, exist_ok=True)
            trainer.save_pretrained(str(save_dir))

    final_dir = Path(cfg.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_pretrained(str(final_dir))
    Path(cfg.output_dir, "ppo_config_used.yaml").write_text(
        "\n".join(f"{k}: {v}" for k, v in asdict(cfg).items()),
        encoding="utf-8",
    )
