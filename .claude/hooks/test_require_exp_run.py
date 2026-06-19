import json, subprocess, sys, pathlib

HOOK = pathlib.Path(__file__).with_name("require_exp_run.py")

def run(command):
    payload = json.dumps({"tool_input": {"command": command}})
    p = subprocess.run([sys.executable, str(HOOK)], input=payload,
                       capture_output=True, text=True)
    return p.returncode

CASES = [
    # (command, expected_exit, label)
    ("ls -la", 0, "unrelated command allowed"),
    ("EXP=04-simcot-pondernet-gammasweep RUN=g0.05-gm3.0-ep5 bash scripts/train_gpt2_gsm8k_pondernet.sh", 0, "valid train allowed"),
    ("EXP=04-x RUN=r CKPT=/c bash scripts/eval_gpt2_gsm8k_pondernet.sh", 0, "valid eval allowed"),
    ("bash scripts/train_gpt2_gsm8k_pondernet.sh", 2, "train without EXP/RUN blocked"),
    ("RUN=r bash scripts/train_gpt2_gsm8k_pondernet.sh", 2, "train missing EXP blocked"),
    ("EXP=gammasweep RUN=r bash scripts/train_gpt2_gsm8k_pondernet.sh", 2, "bad EXP pattern blocked"),
    ("CKPT=/c bash scripts/eval_gpt2_gsm8k_fixedk.sh", 2, "fixedk eval without EXP/RUN blocked"),
]

failed = 0
for cmd, expected, label in CASES:
    got = run(cmd)
    ok = got == expected
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: exit={got} (want {expected})")
    failed += not ok
sys.exit(1 if failed else 0)
