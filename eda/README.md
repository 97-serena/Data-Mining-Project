# Phase 1: Individual Data Exploration

Everyone goes through **all 56 features** independently, so that we all know the data before writing the proposal.

**Deadline: 11 October, 10:00 (German time)**

## What to hand in

Two files per person, uploaded to this `exploration` branch:
- `eda/eda_<yourname>.ipynb`: your notebook with your findings
- `eda/summaries/eda_summary_<yourname>.csv`: your decision table (created by the last cell of the notebook)

## Steps

1. Clone the repo, `git checkout exploration`, and put `cell2celltrain.csv` into `dataset/` (see the main README).
2. Copy `eda_template.ipynb` and rename the copy to `eda_<yourname>.ipynb`. **Never edit the template itself**; it is the shared starting point for everyone.
3. Run your notebook section by section and write your findings under each section.
4. Fill in the decision table at the end, set `NAME` in the last cell, and run it.
5. Clear all outputs and upload your two files, either:
   - **with git:** `git pull`, then `git add` your two files, `git commit -m "EDA <yourname>"`, `git push` (always pull first, otherwise the push is rejected)
   - **on the GitHub website:** switch the branch to `exploration`, open `eda` (and then `eda/summaries` for the CSV) → **Add file** → **Upload files** → commit directly to `exploration`

Only upload your own files; don't change the template, `src/` or other people's files.

## The decision table

One row per feature, saying how you would handle it. `group` and `column` are pre-filled; you fill in:

| Field | Question |
|---|---|
| `issues` | What problems does the column have (missing, placeholders, outliers, skew, redundancy)? |
| `relation_to_churn` | Is it related to churn, how strongly, in which direction? |
| `keep_drop` | keep / drop / unsure |
| `missing_handling` | none / median / mode / placeholder → NaN + indicator / ... |
| `transformation` | none / log1p / clip / binarise / ... |
| `encoding` | numeric / binary 0-1 / ordinal / one-hot / ... |
| `feature_ideas` | New features based on this column (optional) |

All six tables have the same format, so we can merge them and see where we agree and where we disagree. The meeting then focuses on the disagreements, and the agreed table becomes the preprocessing plan for the proposal.
