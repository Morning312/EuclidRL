from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional

from transformers import TrainerCallback, TrainingArguments
from trl import SFTTrainer

from prover.configs import SFTConfig
from prover.data import SupervisedDataCollator, load_jsonl_dataset
from prover.models import get_tokenizer, load_qwen_policy
from prover.models.qwen_policy import PolicyInitConfig


class LogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        if logs:
            print(f"[SFT] step={state.global_step} logs={logs}")


def run_sft(cfg: SFTConfig, config_path: Optional[str] = None) -> None:
    print("Launching SFT with config:", config_path or "defaults")
    policy_cfg = PolicyInitConfig(
        model_name=cfg.model_name,
        load_in_4bit=False,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
    )
    model, tokenizer = load_qwen_policy(policy_cfg)

    train_ds, val_ds = load_jsonl_dataset(cfg.train_file, cfg.val_file)
    collator = SupervisedDataCollator(tokenizer, max_length=cfg.max_seq_length)

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.log_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        bf16=cfg.mixed_precision == "bf16",
        fp16=cfg.mixed_precision == "fp16",
        gradient_checkpointing=cfg.gradient_checkpointing,
        report_to=["none"],
        seed=cfg.seed,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        max_seq_length=cfg.max_seq_length,
        packing=False,
        dataset_text_field=None,
        data_collator=collator,
    )
    trainer.add_callback(LogCallback())
    trainer.save_model(cfg.output_dir + "/init")
    trainer.train()
    trainer.save_state()
    trainer.save_model(cfg.output_dir + "/final")

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.output_dir, "sft_config_used.yaml").write_text(
        "\n".join(f"{k}: {v}" for k, v in asdict(cfg).items())
    )
