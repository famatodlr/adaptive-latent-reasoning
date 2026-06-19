#!/usr/bin/env python3
"""PreToolUse guard: train/eval run scripts must set EXP=<NN-name> and RUN=<id>
inline, so artifacts land under the experiment-scoped layout. Exit 2 blocks."""
import json, re, sys

GUARDED = (
    "train_gpt2_gsm8k_pondernet.sh",
    "eval_gpt2_gsm8k_pondernet.sh",
    "eval_gpt2_gsm8k_fixedk.sh",
)
EXP_RE = re.compile(r"(?:^|\s)EXP=([^\s]+)")
RUN_RE = re.compile(r"(?:^|\s)RUN=([^\s]+)")
EXP_OK = re.compile(r"^[0-9]{2}-[a-z0-9.-]+$")

def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)  # malformed payload → don't block
    cmd = (data.get("tool_input") or {}).get("command", "") or ""
    if not any(g in cmd for g in GUARDED):
        sys.exit(0)
    exp = EXP_RE.search(cmd)
    run = RUN_RE.search(cmd)
    if not exp or not run:
        print("BLOCKED: set EXP=<NN-name> RUN=<run-id> inline before a train/eval "
              "script so artifacts land under the experiment layout "
              "(e.g. EXP=04-simcot-pondernet-gammasweep RUN=g0.05-gm3.0-ep5).",
              file=sys.stderr)
        sys.exit(2)
    if not EXP_OK.match(exp.group(1)):
        print(f"BLOCKED: EXP='{exp.group(1)}' must match ^[0-9]{{2}}-[a-z0-9.-]+$ "
              "(e.g. 04-simcot-pondernet-gammasweep).", file=sys.stderr)
        sys.exit(2)
    sys.exit(0)

if __name__ == "__main__":
    main()
