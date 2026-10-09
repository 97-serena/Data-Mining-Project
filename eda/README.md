# Phase 1: Individual Data Exploration

Everyone goes through **all 56 features** independently, so that we all know the data before writing the proposal.

**Deadline: 11 October, 10:00 (German time)**

## What to hand in

Two files per person, uploaded to this `exploration` branch:
- `eda/eda_<yourname>.ipynb`: your notebook with your findings
- `eda/summaries/eda_summary_<yourname>.csv`: your decision table (created by the last cell of the notebook)

## Steps

1. Clone the repo and `git checkout exploration` (the data is already included in `dataset/`).
2. Copy `eda_template.ipynb` and rename the copy to `eda_<yourname>.ipynb`. **Never edit the template itself**; it is the shared starting point for everyone.
3. Run your notebook section by section and write your findings under each section. In section 8, add at least one analysis of your own (sections 0–7 give everyone the same numbers; this is where your own exploration goes).
4. Fill in the decision table at the end, set `NAME` in the last cell, and run it.
5. Save your notebook **with its outputs** (so the others can see your plots and results without running it) and upload your two files, either:
   - **with git:** `git pull`, then `git add` your two files, `git commit -m "EDA <yourname>"`, `git push` (always pull first, otherwise the push is rejected)
   - **on the GitHub website:** switch the branch to `exploration`, open `eda` (and then `eda/summaries` for the CSV) → **Add file** → **Upload files** → commit directly to `exploration`

Only upload your own files; don't change the template, `src/` or other people's files.

## The decision table

One row per feature, saying how you would handle it. `group` and `column` are pre-filled; you fill in:

| Field | Question |
|---|---|
| `issues` | What problems does the column have (missing, placeholders, outliers, skew, redundancy)? |
| `relation_to_churn` | Is it related to churn, how strongly, in which direction? |
| `keep_drop` | keep / drop / keep (feature selection) — the last one means: keep it in preprocessing, and let feature selection in cross-validation decide whether the model uses it |
| `missing_handling` | none / median / mode / placeholder → NaN + indicator / ... |
| `transformation` | none / log1p / clip / binarise / ... |
| `encoding` | numeric / binary 0-1 / ordinal / one-hot / ... |
| `feature_ideas` | New features based on this column (optional) |

All six tables have the same format, so we can merge them and see where we agree and where we disagree. The meeting then focuses on the disagreements, and the agreed table becomes the preprocessing plan for the proposal.

### Glossary

What the short terms in the table mean. You don't have to use them; plain words are fine too.

**Missing values**

| Term | In plain words | Example |
|---|---|---|
| median | Fill an empty number with the typical (middle) value of that column | an empty `MonthlyRevenue` becomes 48.5 |
| mode | Fill an empty category with the most common category | |
| placeholder → NaN | Some values look real but actually mean "we don't know", so we treat them as empty | `AgeHH1` = 0 (nobody is 0 years old), `HandsetPrice` = "Unknown" |
| unknown flag | An extra yes/no column: "was this value unknown?" We add it because customers with unknown values churn a bit more | `IncomeGroup_unknown` = 1 if the income was 0 |

**Transformation (changing the numbers)**

| Term | In plain words | Example |
|---|---|---|
| clip at 0 | Negative values are impossible here, so turn them into 0 | a monthly charge of −11 becomes 0 |
| clip at p99 | A few customers have extremely high values. We lower them to the value that 99% of customers are below, so these few don't distort the model | `Handsets`: 99% have 7 or fewer, so 24 becomes 7 |
| clip at p1/p99 | Same, but at both ends (extremely low and extremely high values) | `PercChangeMinutes` goes from −3406% to +5192% |
| log1p | Most customers have small values and a few have huge ones. log1p shrinks big numbers much more than small ones, so the gaps become more even. Only matters for simple models like logistic regression; tree models don't need it | `MonthlyMinutes`: 99 → 4.6, 999 → 6.9, 7,359 → 8.9 |
| binarise (> 0) | Replace the count by "used it or not" (1 or 0), because most customers have 0 anyway | `ThreewayCalls`: 0 → 0, 3 → 1 |
| convert to rate | Divide by the number of calls. Someone who calls a lot also has more dropped calls; the rate shows how often it goes wrong *per call* | 10 dropped out of 1,000 calls = 1% |

**Encoding (turning the column into numbers for the model)**

| Term | In plain words | Example |
|---|---|---|
| numeric | Already a number, use it as it is | `MonthsInService` |
| binary 0/1 | Yes → 1, No → 0 | `HandsetWebCapable` |
| ordinal | Categories that have an order become 1, 2, 3, … | `IncomeGroup` 1 (low) … 9 (high) |
| one-hot | Categories without an order: one yes/no column per category | `PrizmCode` → is_Rural, is_Suburban, is_Town, is_Other |
| target encoding | Too many categories for one-hot, so each category is replaced by its churn rate (calculated on the training data only) | each `ServiceArea` city → its churn rate |

**Other**

| Term | In plain words |
|---|---|
| leakage | The column only "knows" the answer because the customer is already leaving (e.g. calls to the retention team). Using it would make the model look better than it really is, so we drop it |
| redundant | The column can be calculated from other columns, so it adds nothing new |
| keep (feature selection) | Not sure if it helps. We keep it for now and let the modelling step test whether it is useful |
