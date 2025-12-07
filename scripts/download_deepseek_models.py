#!/usr/bin/env python3
"""
Download DeepSeek-Prover-V1.5 SFT/RL checkpoints into the local `models/` dir.

Usage:
  python scripts/download_deepseek_models.py
  python scripts/download_deepseek_models.py --models sft --target-dir ./models

Notes:
- Requires `huggingface_hub` (`pip install huggingface_hub`).
- Set `HUGGINGFACE_TOKEN` or pass `--token <hf_token>` if the download needs authentication.
- Base checkpoints are intentionally omitted per request.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from huggingface_hub import snapshot_download

MODEL_MAP = {
    "sft": ("deepseek-ai/DeepSeek-Prover-V1.5-SFT", "DeepSeek-Prover-V1.5-SFT"),
    "rl": ("deepseek-ai/DeepSeek-Prover-V1.5-RL", "DeepSeek-Prover-V1.5-RL"),
}


def download_models(model_keys: Iterable[str], target_dir: Path, token: str | None) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for key in model_keys:
        repo_id, folder_name = MODEL_MAP[key]
        dest = target_dir / folder_name
        print(f"Downloading {repo_id} -> {dest} ...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=dest,
            local_dir_use_symlinks=False,
            resume_download=True,
            token=token,
        )
    print("Done. Checkpoints are stored under:", target_dir.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download DeepSeek-Prover-V1.5 SFT/RL checkpoints.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_MAP.keys(),
        default=list(MODEL_MAP.keys()),
        help="Which checkpoints to download.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path("models"),
        help="Where to place the downloaded checkpoints (git-ignored by default).",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Optional Hugging Face token if your environment requires authentication.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_models(args.models, args.target_dir, args.token)


if __name__ == "__main__":
    main()
