# Spec: `retraining` module

Status: **approved** (2026-10-25)

## Objective

Retrain the model **once a day**, without anyone at the keyboard, on the reference dataset enriched with the
team's validations, and **promote the new model only if it is not worse** than the current one.
This closes the learning loop: validate in the interface → the next day, the model has learned from it.

Requirements:
- **no surprise**: a new model never replaces the current one silently when it is worse; every run leaves a readable trace;
- **no wasted run**: when nothing has changed since the last training (no new or cancelled validation), the job stops;
- **no conflict**: the job never runs twice at the same time, and reads the journal under the same lock as the interface;
- **fair comparison**: both models are scored on the same test ETFs, which the current model never saw during training.

Users: the scheduler of your Windows machine (daily run), you (manual run, run history), the interface
(the new current model is picked up automatically).

Out of scope (v1): notifications (e-mail, Teams), running on a server, deleting old model versions, hyperparameter search beyond the existing `C` grid.

## Daily run

`python -m etf_classifier.retraining.run` chains:

1. **Job lock** (`artifacts/models/retraining.lock`): if another run is in progress, stop with a clear message.
2. **Change detection**: fingerprint of the prepared reference dataset (original CSV + active validations).
   Same fingerprint as the current model's → *skipped* (exit code 0, one line in the history). `--force` retrains anyway.
3. **Training of the candidate** with the same code as `model.train` (same split rules, internal validation, single
   evaluation on its test set). The **Azure embeddings are paid without confirmation** (decision: no budget cap);
   the cost of the run is recorded.
4. **Comparison on the same test set**:
   - the comparison set = the candidate's test ETFs that **the current model did not train on**;
   - both models predict on it; macro F1 is computed **on the categories known by both models**
     (categories created since the last model are reported separately, they cannot be compared);
   - if the comparison set is too small (fewer than 20 ETFs, configurable), the run does not promote and says why.
5. **Promotion rule** (approved):
   - candidate macro F1 ≥ current macro F1 − **0.01** (tolerance, configurable), **and**
   - **no category collapses**: for every category with at least 5 test examples (configurable), the candidate's F1 does
     not drop by more than **0.20** (configurable) compared with the current model, **and**
   - the candidate beats the trivial baseline.
   If everything holds → `promote()`; otherwise the candidate is kept on disk, not promoted, with the reasons.
   No current model yet → the candidate becomes current (same rule as `model.train`).
6. **Trace**:
   - `reports/retraining_<version>.md`: data (ETFs, new validations), cost, both scores, per-category comparison, decision and reasons;
   - `artifacts/models/retraining_history.csv`: one line per run (date, outcome `promoted` / `not_promoted` / `skipped` / `failed`,
     versions, scores, cost, duration);
   - `logs/retraining.log`: technical log (errors with their cause).
7. **Exit codes**: 0 = promoted, skipped or not promoted (normal outcomes); 1 = failure (data unreadable, Azure error…),
   so that the Windows scheduler shows the run as failed.

## Remembering what the current model trained on

To build the comparison set, every model now saves the **names of its training ETFs** (`train_names.json`) and the
**fingerprint of the full dataset** in `metrics.json`. A model trained before this module has neither: the comparison then
uses the candidate's whole test set, and the report says so (the current model may have seen some of these ETFs, which
favours it: the rule stays cautious).

## Windows scheduling

- `scripts/run_daily_retraining.ps1`: goes to the project folder, activates the `.venv`, runs the job, appends the output to `logs/retraining.log`.
- `scripts/register_daily_retraining.ps1 [-Time 10:00]`: creates (or updates) the task *ETF Classifier - daily retraining*
  in the Windows Task Scheduler, for the current user, with **"run as soon as possible after a missed start"**
  (the machine may be off at the scheduled time) and no run if one is already in progress.
- Default time: **10:00** (approved); change it with `-Time`.
- The task runs **only when your Windows session is open** (approved): no administrator rights, no stored password.
- `scripts/unregister_daily_retraining.ps1` removes the task.

## Interface

The **Monitoring** page, *Current model* tab, shows the last retraining runs (date, outcome, scores, reasons) from the history.
The app already picks up a newly promoted model automatically.

## Configuration (`settings.yaml`, new `retraining` section)
```yaml
retraining:
  f1_tolerance: 0.01             # candidate macro F1 >= current macro F1 - tolerance
  category_drop_max: 0.20        # maximum F1 drop allowed for a category ...
  category_min_support: 5        # ... with at least this many test examples
  min_comparison_etfs: 20        # below this, no automatic promotion
  history_path: "artifacts/models/retraining_history.csv"
  log_path: "logs/retraining.log"
```

## Tech Stack
No new Python dependency. PowerShell scripts for the Windows Task Scheduler (built into Windows).

## Commands
```
python -m etf_classifier.retraining.run                 # daily run (what the scheduler calls)
python -m etf_classifier.retraining.run --force         # retrain even if nothing changed
python -m etf_classifier.retraining.run --dry-run       # train and compare, never promote
powershell -ExecutionPolicy Bypass -File scripts\register_daily_retraining.ps1 -Time 10:00
pytest tests/retraining -v
```

## Project structure
```
src/etf_classifier/retraining/
├── __init__.py
├── compare.py        # comparison set, scores on common categories, promotion rule
├── history.py        # run history (CSV) and job lock
└── run.py            # CLI: detection, training, comparison, promotion, report
scripts/
├── run_daily_retraining.ps1
├── register_daily_retraining.ps1
└── unregister_daily_retraining.ps1
tests/retraining/
```
Changes to existing modules: `model.train` (saves `train_names.json` and the dataset fingerprint; training logic
extracted into a function reused by `retraining`, behaviour of the command unchanged); `app` Monitoring page (history of runs).

## Code style
Same as the previous modules. The promotion rule is a pure function, easy to read and test:
```python
def decide(current: Scores, candidate: Scores, cfg: dict) -> Decision:
    """Returns promote=True/False with every reason, never a bare boolean."""
```

## Test strategy
- **Promotion rule** on hand-built scores: within tolerance → promoted; macro F1 too low → refused; one category collapses → refused
  (with its name); candidate below the trivial baseline → refused; comparison set too small → refused.
- **Comparison set**: ETFs seen by the current model in training are excluded; categories created since are excluded from the macro F1 and listed.
- **Change detection**: no new validation → *skipped*, nothing trained; a new or cancelled validation → retraining.
- **End to end on the fixture**: first run without a current model → promoted; second run without change → skipped;
  validation added → retrained and compared; `--dry-run` never promotes.
- **Job lock**: a second run while the lock is held stops with a clear message, without training.
- **History and report** written for every outcome, including a failure (exit code 1).
- **Older model** without `train_names.json`: comparison on the whole test set, mentioned in the report.
- No network call (fake Azure client); the PowerShell scripts are checked by reading them (no Windows in the test environment).

## Boundaries
- **Always:** compare on the same test ETFs; keep every candidate on disk; write the history line, even on failure; read the journal under the lock.
- **Ask first:** changing the promotion rule or its thresholds; deleting model versions; running the job on another machine.
- **Never:** promote a model worse than the trivial baseline; modify the original CSV or the journal; write a secret in the scripts or the logs.

## Success criteria
1. Every day at the scheduled time (or as soon as the machine is on), the job runs; the run appears in the history.
2. Without any new validation, the job stops in a few seconds without training.
3. With new validations, a candidate is trained and compared fairly; it is promoted only if the rule holds, and the report explains the decision.
4. The interface uses the promoted model without being restarted; the Monitoring page shows the last runs.
5. Tests pass without network.

## Decisions made
- Promotion: same test set, macro F1 tolerance of 0.01, no category collapse, better than the trivial baseline.
- Windows Task Scheduler, local machine.
- Azure embeddings paid without confirmation or cap in the daily job (cost recorded in the history and the cost log).
- Every model version is kept (no automatic deletion; about 120 MB per version with embeddings).
- Daily run at **10:00**, only while your Windows session is open; missed runs start as soon as possible.

## Open questions
None.
