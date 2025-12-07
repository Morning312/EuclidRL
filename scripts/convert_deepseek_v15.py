#!/usr/bin/env python
"""
Convert DeepSeek-Prover-V1.5 JSONL files to local SFT format.

Input schema per DeepSeek-Prover-V1.5:
- name, split, informal_prefix, formal_statement, formal_proof, header (unused)

Output schema:
- prompt: informal_prefix + formal_statement + instruction
- completion: Lean proof script (prefixed with a newline)
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple


def build_prompt(informal_prefix: str, formal_statement: str) -> str:
    parts = []
    if informal_prefix:
        parts.append(informal_prefix.strip())
    if formal_statement:
        parts.append(formal_statement.strip())
    parts.append("-- Provide a Lean 4 proof script that discharges the goal above.")
    return "\n".join(parts)


def convert_one_file(path: Path, use_valid_for_train: bool = False) -> Tuple[list, list]:
    train_rows, val_rows = [], []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        split = (row.get("split") or "train").lower()
        prompt = build_prompt(row.get("informal_prefix", ""), row.get("formal_statement", ""))
        completion = "\n" + (row.get("formal_proof") or "").strip()
        out = {"prompt": prompt, "completion": completion}
        if split.startswith("train"):
            train_rows.append(out)
        elif split.startswith(("valid", "val")) or split == "dev":
            if use_valid_for_train:
                train_rows.append(out)
            else:
                val_rows.append(out)
        else:
            continue
    return train_rows, val_rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    rows_list = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows_list))
    print(f"wrote {len(rows_list)} rows to {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="List of DeepSeek-Prover-V1.5 JSONL files (e.g., datasets/minif2f.jsonl ...)",
    )
    ap.add_argument(
        "--train-output",
        default="data/sft/train.jsonl",
        help="Where to write converted train data.",
    )
    ap.add_argument(
        "--val-output",
        default="data/sft/val.jsonl",
        help="Where to write converted val data.",
    )
    ap.add_argument(
        "--use-valid-for-train",
        action="store_true",
        help="If set, treat valid/val/dev splits as train (useful when only eval splits are available).",
    )
    args = ap.parse_args()

    all_train, all_val = [], []
    for p in args.inputs:
        t, v = convert_one_file(Path(p), use_valid_for_train=args.use_valid_for_train)
        all_train.extend(t)
        all_val.extend(v)

    write_jsonl(Path(args.train_output), all_train)
    write_jsonl(Path(args.val_output), all_val)


if __name__ == "__main__":
    main()
