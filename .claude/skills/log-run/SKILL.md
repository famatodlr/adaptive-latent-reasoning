---
name: log-run
description: Record a finished training/eval run into the docs/experiments layer. Use after a run completes to write its per-run .md, update the experiment's runs.md table, and refresh the top-level index headline.
disable-model-invocation: true
---

# log-run

Turn a run's mechanical artifacts into a git-tracked run log.

## Inputs
Ask for (or infer from the path the user gives): `EXP=<NN-exp>`, `RUN=<run-id>`.
The experiment folder `docs/experiments/<NN-exp>/` must already exist (else run `/new-experiment` first).

## Steps

1. **Read the command.** Parse hyperparameters from `outputs/<NN-exp>/<run-id>/command.sh`
   (resolved flags: `--num_train_epochs`, `--learning_rate`, `--per_device_train_batch_size` ×
   `--gradient_accumulation_steps` → eff. batch, `--pondernet_gamma`, `--pondernet_geom_mean`,
   `--max_latent_steps`, `--seed`, warm-start via `SIMCOT_CKPT`, `--data_path`).
2. **Read the metrics.** For each evaluated threshold, read
   `results/<NN-exp>/<run-id>/[thr<x>/]summary.json` (`accuracy_pct`, `avg_steps_used`, `threshold`, `ckpt`).
   If `summary.json` is absent (pre-migration run), ask the user for accuracy/avg-steps or read `eval.log`.
3. **Write `docs/experiments/<NN-exp>/<run-id>.md`** from this template:

   ```markdown
   # <run-id>

   **Experiment:** [<NN-exp>](runs.md)  **Date:** <date>  **Status:** ✅ done | ⚠ known-bad | ✗ failed

   ## Summary
   <one paragraph: what this run is + headline result>

   ## Hyperparameters   _(from `command.sh`)_
   | param | value |
   |-------|-------|
   | method | PonderNet adaptive / fixed-K / baseline |
   | epochs · lr · eff. batch | <e> · <lr> · <bs×accum> |
   | γ · geom_mean · K_max | <γ> · <gm> · <K> |
   | warm-start · seed · data | <full-model/decoder-only> · <seed> · <data file> |

   ## Results   _(from `summary.json`)_
   | checkpoint | threshold | accuracy | avg steps | n_samples |
   |-----------|-----------|----------|-----------|-----------|
   | <ckpt> | <thr> | <acc>% | <steps> | 1 (greedy) |

   ## Artifacts
   - checkpoint: `models/checkpoints/<NN-exp>/<run-id>/`
   - command + log: `outputs/<NN-exp>/<run-id>/{command.sh,train.log}`
   - eval results: `results/<NN-exp>/<run-id>/`

   ## Notes
   <narrative: what was learned, gotchas, anomalies>
   ```

4. **Update `runs.md`.** Add or replace this run's row in `docs/experiments/<NN-exp>/runs.md`:
   `| <run-id> | <key variable> | <best acc>% | <avg steps> | <status emoji> | [detail](<run-id>.md) |`.
5. **Refresh the index.** If this run is the experiment's best, update the experiment's `best result`
   cell in `docs/experiments.md`; if it beats every experiment, update the `**Headline:**` line.
6. **Show the user** the written `<run-id>.md` for a narrative/notes pass before committing.
