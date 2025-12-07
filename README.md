# EuclidRL – Proof Bot (Qwen2.5-Math)

Full SFT + RL pipeline skeleton implementing the workflow from
https://arxiv.org/html/2408.08152v1 on Qwen/Qwen2.5-Math-1.5B.

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# install Lean toolchain (LeanDojo dep)
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
exec $SHELL -l
lean --version
```

Download the model locally (do NOT commit):
```bash
huggingface-cli login  # use your token
huggingface-cli download Qwen/Qwen2.5-Math-1.5B --local-dir models/qwen2.5-math-1.5b
```

## Data layout
- SFT data: `data/sft/train.jsonl`, `data/sft/val.jsonl` with records `{"prompt": "...", "completion": "..."}`.
- RL rollouts: `data/rollouts/train.jsonl` with records `{"prompt": "...", "theorem": "Name.optional"}`.

### Convert DeepSeek-Prover-V1.5 to SFT format
```bash
python scripts/convert_deepseek_v15.py \
  --inputs DeepSeek-Prover-V1.5/datasets/minif2f.jsonl DeepSeek-Prover-V1.5/datasets/proofnet.jsonl \
  --train-output data/sft/train.jsonl \
  --val-output data/sft/val.jsonl
```
This pulls all `split=train` into train and `split=valid` into val; `split=test` is dropped.

## Running SFT
```bash
python scripts/train_sft.py --config prover/configs/sft_default.yaml
```

## Running RL (PPO)
```bash
python scripts/train_rl.py --config prover/configs/ppo_default.yaml \
  --lean-repo https://github.com/leanprover-community/mathlib4.git \
  --lean-commit <commit> \
  --theorems MyTheorem1 MyTheorem2
```

Outputs land in `checkpoints/sft` and `checkpoints/ppo`.
