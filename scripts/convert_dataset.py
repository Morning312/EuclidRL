#!/usr/bin/env python
"""
Convert DeepSeek-Prover-V1 dataset.jsonl to SFT and rollout formats.

Input schema (dataset.jsonl):
- name: theorem identifier (e.g., "thm_0")
- split: always "train" in this dataset
- formal_statement: Lean theorem statement
- goal: proof goal
- header: imports (unused)
- formal_proof: actual Lean proof

Output schemas:
- SFT: {"prompt": "...", "completion": "..."}
- Rollout: {"prompt": "...", "theorem": "..."}
"""

import argparse
import json
import random
from pathlib import Path


def build_prompt(formal_statement: str) -> str:
    """Build prompt from formal statement."""
    parts = [formal_statement.strip()]
    parts.append("-- Provide a Lean 4 proof script that discharges the goal above.")
    return "\n".join(parts)


def load_dataset(path: Path) -> list[dict]:
    """Load all records from JSONL file."""
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def convert_to_sft(record: dict) -> dict:
    """Convert a record to SFT format."""
    prompt = build_prompt(record.get("formal_statement", ""))
    completion = "\n" + (record.get("formal_proof") or "").strip()
    return {"prompt": prompt, "completion": completion}


def convert_to_rollout(record: dict) -> dict:
    """Convert a record to rollout format."""
    prompt = build_prompt(record.get("formal_statement", ""))
    theorem = record.get("name", None)
    return {"prompt": prompt, "theorem": theorem}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write records to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
        encoding="utf-8"
    )
    print(f"Wrote {len(rows)} rows to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert dataset.jsonl to SFT and rollout formats."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/sft/dataset.jsonl",
        help="Path to input dataset.jsonl file.",
    )
    parser.add_argument(
        "--train-output",
        type=str,
        default="data/sft/train.jsonl",
        help="Where to write SFT training data.",
    )
    parser.add_argument(
        "--val-output",
        type=str,
        default="data/sft/val.jsonl",
        help="Where to write SFT validation data.",
    )
    parser.add_argument(
        "--rollout-output",
        type=str,
        default="data/rollouts/train.jsonl",
        help="Where to write rollout data.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum total samples to use (None = use all).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation (default: 0.1 = 10%%).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling and splitting.",
    )
    parser.add_argument(
        "--no-rollouts",
        action="store_true",
        help="Skip generating rollout file.",
    )
    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.input}...")
    records = load_dataset(Path(args.input))
    print(f"Loaded {len(records)} records.")

    # Shuffle for random split
    random.seed(args.seed)
    random.shuffle(records)

    # Limit samples if requested
    if args.max_samples and args.max_samples < len(records):
        records = records[: args.max_samples]
        print(f"Limited to {len(records)} samples.")

    # Split into train/val
    val_size = int(len(records) * args.val_ratio)
    train_size = len(records) - val_size

    train_records = records[:train_size]
    val_records = records[train_size:]

    print(f"Split: {len(train_records)} train, {len(val_records)} val")

    # Convert to SFT format
    train_sft = [convert_to_sft(r) for r in train_records]
    val_sft = [convert_to_sft(r) for r in val_records]

    # Write SFT files
    write_jsonl(Path(args.train_output), train_sft)
    write_jsonl(Path(args.val_output), val_sft)

    # Generate rollout file (from train set only)
    if not args.no_rollouts:
        rollouts = [convert_to_rollout(r) for r in train_records]
        write_jsonl(Path(args.rollout_output), rollouts)

    print("\nDone! Summary:")
    print(f"  Train: {len(train_sft)} examples -> {args.train_output}")
    print(f"  Val:   {len(val_sft)} examples -> {args.val_output}")
    if not args.no_rollouts:
        print(f"  Rollouts: {len(rollouts)} examples -> {args.rollout_output}")


if __name__ == "__main__":
    main()
