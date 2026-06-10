# Repo Professionalization — Design Spec

**Date:** 2026-06-10
**Branch:** `pondernet`
**Author:** brainstormed with Claude
**Status:** Draft for review

## Goal

Make the `adaptive-latent-reasoning` repo more professional and navigable, focused on five concrete pain points identified during development:

1. No clear training pipeline / workflow description.
2. Unclear model names in `models/`.
3. Two model directories (`models/` and `pondernet/models/`); the decoder lives in the latter and should move.
4. Messy contents in `models/`, `outputs/`, `results/`.
5. Unclear PonderNet parameter / variable names.

This is a single, cohesive structure-and-documentation pass. It explicitly does **not** change training or loss logic, and does **not** retrain anything.

## Guiding decisions (from brainstorming)

- `models/` is **split by provenance**: downloaded/derived vs. our trained runs.
- Variable cleanup is **safe-only** (Option 1): rename non-checkpoint identifiers + add a glossary; do **not** rename `nn.Module` attributes that are baked into checkpoint `state_dict` keys.
- Pipeline is documented as a **prose doc + diagram** (no script-behavior/config-system changes beyond path updates).
- One **run-id** names an experiment identically across `models/checkpoints/`, `outputs/`, and `results/`, backed by a `runs.md` manifest.
- The warm-start **decoder is gitignored** (no longer git-tracked); it is reproducible via `scripts/fetch_simcot_decoder.py`.
- Existing artifacts are **migrated** to the new convention; a small set of genuinely-dead artifacts is **deleted** (see Cleanup), and one buried checkpoint is **rescued**.

---

## 1. Target directory structure

```
models/
  pretrained/                       # downloaded / derived, read-only, GITIGNORED
    gpt2/
    simcot-gpt2-codi/
    simcot-gpt2-coconut/
    simcot-gpt2-decoder/            # moved from pondernet/models/, now gitignored
  checkpoints/                      # our trained runs, one subdir per run-id, GITIGNORED
    <run-id>/

outputs/                            # TensorBoard logs, one subdir per run-id, GITIGNORED
  <run-id>/

results/                            # eval outputs, one subdir per run-id, GITIGNORED
  <run-id>/

docs/
  pipeline.md                       # NEW — training workflow narrative + diagram
  parameters.md                     # NEW — parameter/variable glossary + CLI reference
  runs.md                           # NEW — run manifest table
  methods-comparison.md             # existing
  superpowers/specs/                # this spec lives here
```

The three run-scoped trees (`models/checkpoints/`, `outputs/`, `results/`) are **parallel**: the same `<run-id>` names the same experiment in all three. The current `results/{fixedk,pondernet}/` method-grouping is **flattened** — method is encoded in the run-id instead, keeping the trees mirror-able.

`pondernet/models/` is removed once the decoder moves.

## 2. Naming conventions

### Pretrained models (lowercase-kebab, descriptive)

| Current | New |
|---|---|
| `models/gpt2` | `models/pretrained/gpt2` |
| `models/SIM_COT-GPT2-CODI` | `models/pretrained/simcot-gpt2-codi` |
| `models/SIM_COT-GPT2-Coconut` | `models/pretrained/simcot-gpt2-coconut` |
| `pondernet/models/simcot_gpt2_decoder` | `models/pretrained/simcot-gpt2-decoder` |

### Run-id scheme

`<base>-<method>[-<key-hparams>]`, where key-hparams are the ones that *distinguish* runs (K / latent steps, geom mean, learning rate, epochs). Examples:

- `simcot-baseline-k6`
- `simcot-fixedk-k6-lr1e4`
- `simcot-pondernet-lr1e4`
- `simcot-pondernet-joint-ep40`

**Inference threshold is not a training distinction.** `thr0.8` vs `thr0.9` are the same trained model evaluated twice, so they become threshold sub-variants under a single run-id's `results/` dir (e.g. `results/<run-id>/thr0.8/`, `thr0.9/`), not separate runs.

## 3. `.gitignore` changes

The decoder is currently the only git-tracked model artifact (via `!pondernet/models/simcot_gpt2_decoder/**` exceptions). Per decision, it becomes gitignored like everything else:

- `git rm --cached -r pondernet/models/simcot_gpt2_decoder` (stops tracking; file stays on disk to be moved).
- Remove the `!pondernet/models/...` un-ignore lines (current `.gitignore` lines 11–13).
- Keep `models/`, `outputs/`, `results/` ignored uniformly (no exceptions).
- `pipeline.md` documents how to (re)obtain the decoder: `python scripts/fetch_simcot_decoder.py --out ../models/pretrained/simcot-gpt2-decoder`.

## 4. Decoder move

- `git mv` is not applicable (untracked after step 3); move on disk to `models/pretrained/simcot-gpt2-decoder/`.
- Update `scripts/fetch_simcot_decoder.py` default `--out` to the new path.
- Update script env-var defaults that reference pretrained models:
  `DECODER_PATH`, `GPT2_PATH`, `SIMCOT_CKPT` in `pondernet/scripts/*.sh`
  (currently `./models/simcot_gpt2_decoder`, `../models/SIM_COT-GPT2-CODI`, etc.) → new `../models/pretrained/...` paths.
- Delete the now-empty `pondernet/models/`.

## 5. Variable-name clarity (safe-only)

### Rename in code (NOT in any checkpoint — safe)

| Current | Proposed | Kind | Notes |
|---|---|---|---|
| `num_latent` | `max_latent_steps` | CLI flag + locals | Rename the flag and update **all in-repo scripts** that pass it in the same commit (all callers are in-repo, so no alias needed). Hard upper bound K_max, not the actual adaptive count. |
| `forward_idx` | `step_idx` | local loop counter | indexes latent steps |
| `explain_embds_list` | `step_token_ids` | local | holds token IDs, not embeddings |
| `latent_embd` | `latent_hidden` | local | it is a hidden state, not a token embedding |

### Keep as-is + document in `docs/parameters.md`

These are `nn.Module` attributes whose names appear in saved checkpoint `state_dict` keys; renaming them would break loading of existing checkpoints (and the upstream SIM-CoT checkpoint we warm-start from). Each gets a glossary entry + a one-line inline comment at its definition:

- `codi` (the LoRA backbone), `decoder` (auxiliary step decoder), `prj` (projection), `halt_head` (halting head)
- `pj_in` / `pj_out` — **correction from initial triage:** these are `nn.Module` attributes (`self.pj_in = nn.Linear(...)`, `self.pj_out = ...` at `pondernet/src/model.py:335,342`), so they live in checkpoint keys too and are **kept**, not renamed.

### `docs/parameters.md` contents

- Full CLI-flag reference: every train/eval flag, its default, and what it controls.
- Warm-start recipe explainer: decoder-only vs full-model; the `model_name_or_path` ≠ CODI-checkpoint trap; what `simcot_ckpt` / `decoder_path` do and how the sentinel checks guard them.
- Glossary for the kept-but-jargon names above and the loss terms (`l_pondernet`, `kl_geom`, `explain_loss`, distill/ref losses).

## 6. Migration & cleanup

Nothing is deleted before the `runs.md` mapping table is approved. Because `models/`, `outputs/`, `results/` are gitignored, deletions are **not git-recoverable** — they are done deliberately and only for the items below.

### Migrate (rename to run-id)

| Old location(s) | New run-id | Trees |
|---|---|---|
| `models/simcot_joint_ep40`, `outputs/simcot_joint_ep40`, `results/pondernet/simcot_joint_ep40` | `simcot-pondernet-joint-ep40` | checkpoints / outputs / results — flag **known-bad (~19% vs 39.5%)** |
| `models/halt_head_gpt2_ep40`, `outputs/halt_head_gpt2_ep40` | `simcot-pondernet-halthead-ep40` | checkpoints / outputs |
| `results/fixedk/lr1e4_k6` | `simcot-fixedk-k6-lr1e4` | results |
| `results/fixedk/simcot_baseline_k6` | `simcot-baseline-k6` | results |
| `results/pondernet/lr1e4_thr0.8` + `lr1e4_thr0.9` | `simcot-pondernet-lr1e4` (as `thr0.8/`, `thr0.9/` sub-variants) | results |
| `outputs/simcot_warmstart_lr1e4`, `results/pondernet/simcot_warmstart_lr1e4` | `simcot-pondernet-warmstart-lr1e4` | outputs / results |

### Rescue (buried real checkpoint)

`outputs/pondernet/.../ep_40/lr_0.0001/seed_42/` contains the only copy of the trained **lr1e4 PonderNet model** (final `pytorch_model.bin`, 578 MB) — the source of the kept `lr1e4_thr*` results. It was misfiled into `outputs/` because `SAVE_DIR`/`LOG_DIR` pointed there.

- Move the **final** model + tokenizer/config files → `models/checkpoints/simcot-pondernet-lr1e4/`.
- Keep its TB logs → `outputs/simcot-pondernet-lr1e4/`.
- **Prune** the redundant resume state: `checkpoint-4563/`, `checkpoint-4680/`, `optimizer.pt` (~1.3 GB reclaimed; only needed to resume training).
- Remove the empty `outputs/pondernet/.../lr_0.003/` and the now-empty `outputs/pondernet/` shell.

### Delete (genuinely dead — verified)

| Target | Why safe |
|---|---|
| `outputs/simcot_joint_warmstart/` | crashed full-model warm-start run; ends in a traceback at `accelerator.backward` (`train.py:510`); no saved model. Recorded in manifest as FAILED. |
| `outputs/pondernet/.../lr_0.003/` (empty) | produced no files |
| rescued run's `checkpoint-45xx/`, `checkpoint-46xx/`, `optimizer.pt` | training-resume state, redundant once final model is kept |
| `results/pondernet/gsm8k.json`, `results/pondernet/gsm8k_pondernet_detail.json` | stray; `gsm8k.json` is **byte-identical** to `lr1e4_thr0.8/gsm8k.json` (verified) — a duplicate |

### `runs.md` manifest

One row per run-id, columns: `run-id | date | method | key hparams | result (accuracy) | checkpoint? | notes`.
- `date` seeded from TensorBoard event-file Unix timestamps.
- `result` accuracies read from each run's `gsm8k.json` during table construction.
- FAILED and known-bad runs are listed with their flag so history stays traceable.
- The full table is presented for user approval **before** any move/delete is executed.

## 7. Pipeline doc (`docs/pipeline.md`)

Prose walkthrough + a Mermaid flow diagram covering:

1. Acquire base + warm-start artifacts (incl. `fetch_simcot_decoder.py`).
2. Choose warm-start recipe (full-model via `simcot_ckpt` vs decoder-only via `decoder_path`); the `model_name_or_path` trap.
3. Train (`scripts/train_gpt2_gsm8k_pondernet.sh`), naming the run via the run-id convention.
4. Eval (`scripts/eval_gpt2_gsm8k_{fixedk,pondernet}.sh`).
5. Read `results/<run-id>/` and record the run in `runs.md`.

Cross-links to `parameters.md` (glossary) and `runs.md` (history). Avoids TensorBoard-only phrasing so a future logging swap reads cleanly.

## 8. Future: MLflow compatibility (not implemented now)

The design is intentionally MLflow-ready so a later integration needs no rework:

- Trainer is `transformers.Trainer` (`CustomTrainer`), which has native MLflow support via `report_to="mlflow"` — additive to the current TensorBoard logging.
- The run-id is the single source of truth → use it as the MLflow run name/tag so local dirs and MLflow runs stay 1:1.
- `parameters.md` defines the param schema for `mlflow.log_params()`.
- `runs.md` is a manual stand-in for the MLflow tracking UI; they coexist, and MLflow can later supersede it with the same columns.
- Reserve `mlruns/` in `.gitignore` for when it lands.

## 9. Scope

**In scope:** directory restructure; pretrained renames; decoder move + gitignore; run migration, rescue, and targeted cleanup; `runs.md` manifest; 4 safe code renames + inline comments; `parameters.md` glossary; `pipeline.md`; MLflow-readiness notes/gitignore reservation.

**Out of scope (non-goals):**
- Renaming checkpoint-bound module attrs (`codi`, `decoder`, `prj`, `halt_head`, `pj_in`, `pj_out`) — would require a state_dict remap shim; deferred.
- Config-driven run files / changes to the script argument system beyond path updates.
- Any change to training, loss, or inference logic.
- Retraining the known-bad run or the failed warm-start.
- Implementing MLflow.

## 10. Sequencing (for the implementation plan)

Each step is independently verifiable:

1. Docs scaffolding: `parameters.md` glossary + CLI reference.
2. Safe code renames (4 identifiers) + update in-repo scripts; verify training/eval still launches.
3. Directory restructure: `models/pretrained/` + decoder move + `.gitignore` update + script path updates.
4. Build `runs.md` mapping table → **user approval gate** → execute migration, rescue, and deletions.
5. `pipeline.md` (depends on final paths/run-ids) + MLflow-readiness gitignore reservation.

## 11. Verification

- **Code renames:** load `simcot_joint_ep40` (now `simcot-pondernet-joint-ep40`) before and after the change, run identical input, compare logits — must be identical (renames touch no checkpoint keys).
- **Decoder move:** `fetch_simcot_decoder.py --out <new path>` reproduces the decoder; eval using the new `DECODER_PATH` runs.
- **Migration/cleanup:** executed only after the `runs.md` table is approved; rescued checkpoint loads and reproduces a kept `lr1e4` result.
- **No silent loss:** every deleted item is listed in the manifest with its reason.

## Open items requiring user input during implementation

- Final `runs.md` table values (dates, accuracies) and confirmation of the `outputs/simcot_joint_warmstart` and stray-json dispositions at execution time.
- Confirm the run-id token spellings (e.g. `lr1e4`, `geom3`, `k6`) before mass-renaming.
