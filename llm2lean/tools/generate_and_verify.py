import os, re, subprocess, time, uuid, csv, sys, json, shlex

# ---------- CONFIG ----------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GEN_DIR = os.path.join(PROJECT_ROOT, "Gen")
PROMPT_PATH = os.path.join(PROJECT_ROOT, "prompts", "lean_gen_prompt.txt")

# Choose your LLM API (example: OpenAI)
# Set env vars before running:
#   export OPENAI_API_KEY=...
MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")  # change as you like
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Simple shell helper
def run(cmd, cwd=None, timeout=120):
    print(f"$ {cmd}")
    proc = subprocess.run(cmd, cwd=cwd, shell=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=timeout, text=True)
    return proc.returncode, proc.stdout, proc.stderr

def read_prompt():
    with open(PROMPT_PATH, "r") as f:
        return f.read()

def call_llm(system_prompt, user_prompt):
    # ---- Replace with your preferred LLM client/library. Minimal HTTP example shown. ----
    import urllib.request, ssl
    import urllib.error
    endpoint = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": 0.6,
        "max_tokens": 1200,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(endpoint, data=data,
                                 headers={
                                     "Content-Type": "application/json",
                                     "Authorization": f"Bearer {OPENAI_API_KEY}"
                                 })
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
            resp_data = resp.read()
        j = json.loads(resp_data.decode("utf-8"))
        text = j["choices"][0]["message"]["content"]
        return text
    except urllib.error.HTTPError as e:
        print("HTTPError:", e.read().decode(), file=sys.stderr)
        raise
    except Exception as e:
        print("LLM error:", e, file=sys.stderr)
        raise

def extract_lean_block(text):
    # Extract the first ```lean ... ``` block; fallback: any ``` ... ```
    m = re.search(r"```lean(.*?)(```)", text, flags=re.S)
    if not m:
        m = re.search(r"```(.*?)(```)", text, flags=re.S)
    if m:
        return m.group(1).strip()
    # if user returned raw Lean without fences
    return text.strip()

def write_lean_file(code, fname=None):
    os.makedirs(GEN_DIR, exist_ok=True)
    slug = fname or f"Auto_{uuid.uuid4().hex[:8]}"
    path = os.path.join(GEN_DIR, f"{slug}.lean")
    with open(path, "w") as f:
        f.write(code + "\n")
    return path

def has_sorry_or_admit(path):
    txt = open(path, "r", encoding="utf-8").read()
    return bool(re.search(r"\b(sorry|admit)\b", txt))

def try_tactic_rewrites(path, tactics):
    # Replace a trailing `by` followed by sorry with a tactic, or replace `sorry` alone.
    src = open(path, "r", encoding="utf-8").read()
    for t in tactics:
        trial = re.sub(r"\bby\s+sorry\b", f"by {t}", src)
        trial = re.sub(r"\bsorry\b", t, trial)
        trial_path = path.replace(".lean", f".{t}.lean")
        with open(trial_path, "w") as f:
            f.write(trial)
        code, out, err = run("lake build", cwd=PROJECT_ROOT, timeout=300)
        if code == 0:
            return trial_path, t
    return None, None

def main():
    os.makedirs(os.path.join(PROJECT_ROOT, "Logs"), exist_ok=True)
    log_csv = os.path.join(PROJECT_ROOT, "Logs", "runs.csv")
    if not os.path.exists(log_csv):
        with open(log_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id","file","built","proved","tactic","stdout_len","stderr_len","seconds"])

    # Make sure deps are present (no-op if already set)
    run("lake update", cwd=PROJECT_ROOT, timeout=600)

    system_prompt = "You are generating small, valid Lean 4 theorems for mathlib4 projects."
    user_prompt = read_prompt()

    t0 = time.time()
    text = call_llm(system_prompt, user_prompt)
    lean_code = extract_lean_block(text)

    path = write_lean_file(lean_code)
    print(f"Wrote: {path}")

    # First build attempt
    code, out, err = run("lake build", cwd=PROJECT_ROOT, timeout=600)
    built = (code == 0)
    proved = built and (not has_sorry_or_admit(path))
    tactic_used = ""

    # If it built but contains sorry/admit, try quick auto-prove pass
    if built and not proved:
        trial_path, tactic = try_tactic_rewrites(path, ["aesop", "linarith", "norm_num"])
        if trial_path:
            path = trial_path
            built = True
            tactic_used = tactic
            proved = not has_sorry_or_admit(path)  # should be true if it compiled

    # Log results
    rid = uuid.uuid4().hex[:8]
    with open(log_csv, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([rid, os.path.relpath(path, PROJECT_ROOT), built, proved, tactic_used, len(out), len(err), round(time.time()-t0,2)])

    print("\n=== RESULT ===")
    print("file:   ", path)
    print("built:  ", built)
    print("proved: ", proved)
    print("tactic: ", tactic_used or "(none)")
    if code != 0:
        print("\n--- build stdout ---\n", out[:2000])
        print("\n--- build stderr ---\n", err[:2000])

if __name__ == "__main__":
    main()