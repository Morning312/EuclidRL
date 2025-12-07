# EuclidRL – DeepSeek Prover V1 RL

## Quickstart
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# install Lean toolchain
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
exec $SHELL -l
lean --version

# download model locally (do NOT commit)
huggingface-cli login   # use your read-only token
huggingface-cli download deepseek-ai/DeepSeek-Prover-V1 \
  --local-dir models/deepseek-prover-v1

# run a quick generation
python scripts/inference_mps.py
