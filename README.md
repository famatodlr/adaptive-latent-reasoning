# adaptive-latent-reasoning

NLP final project — adaptive latent chain-of-thought reasoning.

We extend [SIM-CoT](https://arxiv.org/pdf/2509.20317) (ICLR 2026) to make the number of implicit reasoning steps **variable at inference time**, rather than fixed.

## Approach

Two parallel strategies under active development:

| Subgroup | Strategy | Branch |
|----------|----------|--------|
| 1 | PonderNet-style adaptive halting over latent tokens | `develop-c` |
| 2 | Fixed-k sweep across backends (Coconut / CODI) | `option-a` |

## Structure

```
baselines/           # Reference implementations (Coconut, CODI, SIM-CoT)
Option-A/            # Subgroup 2 — k-sweep pipeline
docs/papers/         # Paper summaries for context
docs/exploration.md  # Research notes and experiment log
```

## Setup

```bash
uv sync
# or
pip install -r baselines/CODI/requirements.txt
```

## Context for contributors

See `CLAUDE.md` for a full briefing on repo layout, subgroup strategies, and how to run the baselines.
