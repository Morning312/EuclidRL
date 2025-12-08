#!/usr/bin/env python
"""
Demo script for running inference with a trained model.

Usage:
    # Use base Qwen model (before training)
    python scripts/inference_demo.py

    # Use SFT-trained model
    python scripts/inference_demo.py --model checkpoints/sft/final

    # Use PPO-trained model
    python scripts/inference_demo.py --model checkpoints/ppo/final
"""

import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_MODEL = "Qwen/Qwen2.5-Math-1.5B"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference demo.")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model path or HuggingFace ID (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )

    # Example Lean proof goal
    goal_text = "n : ℕ ⊢ gcd n n = n"
    prompt = (
        f"theorem example (n : ℕ) : Nat.gcd n n = n := by\n"
        f"-- Provide a Lean 4 proof script that discharges the goal above."
    )

    print(f"\nPrompt:\n{prompt}\n")
    print("Generating proof...")

    inputs = tok(prompt, return_tensors="pt").to(mdl.device)
    gen = mdl.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    output = tok.decode(gen[0], skip_special_tokens=True)
    print(f"\nGenerated output:\n{output}")


if __name__ == "__main__":
    main()
