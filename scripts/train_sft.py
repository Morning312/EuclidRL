#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prover.configs import load_sft_config
from prover.training import run_sft


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SFT for Qwen proof bot.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_sft_config(args.config)
    run_sft(cfg, args.config)


if __name__ == "__main__":
    main()
