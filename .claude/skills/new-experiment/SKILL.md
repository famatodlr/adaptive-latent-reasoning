---
name: new-experiment
description: Scaffold a new experiment folder under docs/experiments/. Use when starting a new training/eval investigation (a sweep, ablation, or one-off run) that needs a numbered experiment before any runs are logged.
disable-model-invocation: true
---

# new-experiment

Create the git-tracked scaffolding for a new experiment.

## Steps

1. **Pick the number.** List `docs/experiments/`; `NN` = (highest existing `NN-` prefix) + 1, zero-padded to 2 digits. If none exist, `NN=01`.
2. **Pick the name.** Ask the user for a short kebab name including the method prefix (`simcot` always; add `pondernet` when the method applies). Full dir = `<NN>-<name>` and must match `^[0-9]{2}-[a-z0-9.-]+$`.
3. **Get the narrative.** Ask the user: what's the question/hypothesis? what's varied vs held fixed?
4. **Create `docs/experiments/<NN-exp>/experiment.md`** from this template (fill from the answers; leave Findings as a stub to update later):

   ```markdown
   # <NN>: <experiment title>

   **Status:** active   **Dates:** <today> → —

   ## What's being tested
   <hypothesis / question; what's varied vs held fixed>

   ## Setup
   <config shared by all runs: warm-start, data, eff. batch, K_max; what differs per run>

   ## Findings
   _(pending — update when runs complete)_

   See [runs.md](runs.md) for the run table · artifacts under `<dir>/<NN-exp>/`.
   ```

5. **Create `docs/experiments/<NN-exp>/runs.md`** (empty table):

   ```markdown
   # <NN-exp> — Runs

   See [experiment.md](experiment.md) for what's being tested.

   | run | key variable | best accuracy | avg steps | status | detail |
   |-----|-------------|--------------|-----------|--------|--------|
   ```

6. **Add a row to `docs/experiments.md`** under the **main index table** (the `| # | experiment | what it tested | best result | status |` one — *not* the "Active / deferred" table): `| <NN> | [<name>](experiments/<NN-exp>/experiment.md) | <one-line what> | — | active |`.
7. **Tell the user** the exact `EXP=<NN-exp>` string to use when launching runs, and to log finished runs with `/log-run`.
