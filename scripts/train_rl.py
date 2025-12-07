#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prover.configs import load_ppo_config
from prover.env import LeanProofEnv
from prover.training import run_ppo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PPO RL fine-tuning for Qwen proof bot.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    parser.add_argument("--lean-repo", type=str, default=None, help="Lean repo URL for LeanDojo.")
    parser.add_argument("--lean-commit", type=str, default=None, help="Commit hash for LeanDojo.")
    parser.add_argument("--theorems", type=str, nargs="*", default=None, help="Theorems to practice.")
    args = parser.parse_args()

    cfg = load_ppo_config(args.config)
    env = None
    if args.lean_repo and args.lean_commit and args.theorems:
        env = LeanProofEnv(args.lean_repo, args.lean_commit, args.theorems)

    run_ppo(cfg, args.config, env)


if __name__ == "__main__":
    main()
