---
name: analyze-experiment
description: Orient within an existing experiment — load its docs, reconcile them against on-disk artifacts, synthesize the across-run frontier, and flag eval/doc gaps. Read-only (edits nothing). Use when opening a session to work over an experiment that already has a docs/experiments/<NN-exp>/ folder.
disable-model-invocation: true
---

# analyze-experiment

Read-only orientation for an existing experiment. Loads context, reconciles docs vs
disk, synthesizes the frontier, and prints a punch-list of gaps. **Edits nothing** —
act on the punch-list with `/log-run` or by hand.

## Input
`EXP=<NN-exp>` (the migrated experiment dir name, e.g. `04-simcot-pondernet-gammasweep`).
`docs/experiments/<NN-exp>/` **must already exist** — if it doesn't, stop and tell the
user to run `/new-experiment` first (this skill does not handle flat/un-migrated sweeps).

## Steps

1. **Load docs.** Read `docs/experiments/<NN-exp>/experiment.md` and `runs.md`. Note the
   declared runs (rows in `runs.md`) and their recorded best-accuracy / status.
2. **Scan disk.** List run dirs under each of `models/checkpoints/<NN-exp>/`,
   `outputs/<NN-exp>/`, `results/<NN-exp>/`. The union of dir names is the **run-id set**.
3. **Read metrics.** For each run, read every
   `results/<NN-exp>/<run>/[thr<x>/]summary.json` and pull `accuracy_pct`,
   `avg_steps_used`, `threshold`, `ckpt`. (A run may have several — one per threshold.)
4. **Synthesize (orientation).** Print a compact summary: the experiment, the number of
   runs, the per-run accuracy-vs-steps frontier (best operating point at thr=0.8, plus the
   low-step thr=0.5 point where available), and the **current best** run + operating point.
5. **Reconcile — flag exactly these three gaps** (and nothing else):
   - **(a) Trained but not evaluated** — a run has a checkpoint dir
     (`models/checkpoints/<NN-exp>/<run>/`) but **no** `results/<NN-exp>/<run>/`
     (or that dir has no `summary.json` anywhere under it). → "eval pending".
   - **(b) Missing / non-machine-readable metrics** — `results/<NN-exp>/<run>/` has an
     `eval.log` but **no** `summary.json` (a pre-enrichment run). → "re-eval to emit
     `summary.json`, or read the numbers from `eval.log`; then `/log-run`".
   - **(c) `runs.md` ↔ disk mismatch** — a `runs.md` row whose run-id has **no dir on
     disk**, OR whose recorded best-accuracy **diverges** from the matching
     `summary.json` (tolerance ~0.1pp). → "reconcile the row".
6. **Output the report.** One read-only chat message:
   - header: `<NN-exp> — <N> runs` and `best: <run> <acc>% @ <steps> steps (thr<x>)`,
   - a short frontier table/line,
   - a `GAPS` section listing each finding with its one-line suggested follow-up
     (or "none" for a clean category).

   Do **not** edit any file. If the user then wants the docs updated, that's `/log-run`
   (per run) or a manual Findings/headline edit.

## Out of scope (by design)
- Any file edits — Findings, `runs.md` rows, and the index headline are left to `/log-run`
  or the user.
- Undocumented-run detection (a disk run with no `runs.md` row) and unswept-threshold
  detection — not flagged.
- Flat/un-migrated experiments (no `docs/experiments/<NN-exp>/` folder).
