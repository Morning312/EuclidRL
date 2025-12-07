from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "models/deepseek-prover-v1"  # or "deepseek-ai/DeepSeek-Prover-V1"

tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype="auto"
)

# A tiny prompt pattern: give a Lean goal; ask for a tactic
goal_text = "n : ℕ ⊢ gcd n n = n"
prompt = (
    "You are a Lean 4 prover. Given the current goal, output the NEXT TACTIC ONLY.\n"
    f"STATE:\n{goal_text}\n"
    "TACTIC:\n"
)

inputs = tok(prompt, return_tensors="pt").to(mdl.device)
gen = mdl.generate(
    **inputs,
    max_new_tokens=64,
    do_sample=True, top_p=0.9, temperature=0.7,
    eos_token_id=tok.eos_token_id,
)
print(tok.decode(gen[0], skip_special_tokens=True))