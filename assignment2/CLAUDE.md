# CLAUDE.md — Instructions for QBUS2820 Assignment 2

You are helping me (Sherry, a university student) complete QBUS2820 Assignment 2: **forecasting quarterly CPI**.
This file applies to all work inside `assignment2/`. For this folder it **replaces** the Assignment 1 rules in the
repo-root `CLAUDE.md`. Do not edit or move any Assignment 1 files.

Read `spec/QBUS2820_S2_2026_Assignment2.pdf` first.

**The most important rule:** the code must look like the code I learned in my tutorials.
My time-series tutorial notebooks (Week 8 onwards) are the style guide and the method whitelist. Imitate them;
do not invent your own style.

---

## 0. The task in one paragraph

`CPI_train.csv` has 100 quarterly CPI values (columns `Quarter`, `CPI`; `Mar-00` to `Dec-24`).
Forecast the next 8 quarters (`Mar-25` to `Dec-26`) — a **multiple-step-ahead** forecast made once from the end of
the training data (h = 1, …, 8). There are no covariates. Marking: 11 marks for test MSE (relative to the best
submission) and implementation; 14 marks for notebook presentation and the data-analysis procedure.

---

## 1. Folder layout (expected)

```
assignment2/
├── CLAUDE.md                        ← this file
├── spec/QBUS2820_S2_2026_Assignment2.pdf
├── CPI_train.csv
├── tutorials/                       ← time-series tutorial notebooks + data (Week 8 onwards)
├── lectures/                        ← time-series lecture PDFs
└── (outputs you create)
    ├── tutorial_patterns.md
    ├── SID_implementation.ipynb
    ├── CPI_forecast.csv
    ├── results_log.md
    ├── report_notes.md
    └── ai_use_log.md
```

If `tutorials/` has no time-series notebooks, **stop and ask me to add the files**. Do not continue on generic code.

---

## 2. Work in phases. Stop at each checkpoint and wait for my approval.

### Phase 0 — Learn my time-series tutorial code (no modelling yet)

Read every file in `tutorials/` and `lectures/`. Then write `tutorial_patterns.md` with these sections:

1. **Libraries and imports.** Exactly which libraries appear (e.g. `pandas`, `numpy`, `matplotlib`, `statsmodels`),
   with the import aliases the tutorials use.
2. **Method → tutorial code map.** For each method, record the tutorial file and cell, the exact functions/classes
   used, and a short code excerpt. Map at least whichever of these appear: time index / date parsing, time plots,
   classical decomposition (additive/multiplicative), moving averages, ACF/PACF, naive and seasonal-naive forecasts,
   simple/Holt/Holt-Winters exponential smoothing (ETS), ARIMA/SARIMA, regression with trend and seasonal dummies,
   Box-Cox / log transforms, differencing and stationarity tests, train/validation split for time series,
   forecast evaluation (MSE), forecast intervals, residual diagnostics (e.g. Ljung-Box).
3. **Coding style.** Variable naming, how a series is indexed, how models are fitted and forecast, how MSE is
   computed, how plots are made (size, titles, labels), how markdown and comments are written, loop style for tuning.
4. **Gaps.** Methods from the lectures that have no tutorial code.

**Checkpoint 0:** show me `tutorial_patterns.md` and a short summary. Wait for my OK.

### Phase 1 — Data understanding and EDA

In the notebook, in tutorial style:
- load the data, convert `Quarter` (e.g. `Mar-00`) to a proper quarterly time index, and check length, missing values,
  duplicates and that the quarters are consecutive;
- summary statistics (4 d.p.);
- time plot of CPI;
- seasonal plot / mean by quarter (Mar, Jun, Sep, Dec);
- decomposition (trend, seasonal, remainder), and a check of whether the seasonal swing grows with the level
  (additive vs multiplicative);
- trend behaviour over time (e.g. growth rate by sub-period), and any structural changes (e.g. 2020–2022);
- ACF/PACF of the series and of its (seasonally) differenced versions.

Every figure needs an informative title, axis labels and a legend where relevant.

**Checkpoint 1:** summarise the EDA findings in 5–10 bullet points and propose candidate models, each with its
justification. Wait for my OK.

### Phase 2 — Modelling

**Validation scheme (fixed for every model):** time-ordered only — **never shuffle**.
Use rolling-origin (expanding-window) evaluation that mimics the real task: at each origin, fit on data up to that
origin and forecast the next 8 quarters; average the squared errors over h = 1, …, 8 and over origins.
Use the same set of origins for every model (or whatever validation scheme the tutorials use — confirm in Phase 0).

Candidate models (final list depends on Phase 0 and Checkpoint 1):

| ID | Model |
|---|---|
| F0 | Benchmarks: naive, seasonal naive, drift / seasonal naive with drift |
| F1 | Regression on time trend + quarterly dummies |
| F2 | Exponential smoothing: Holt-Winters / ETS variants (additive vs multiplicative season, damped vs not) |
| F3 | (S)ARIMA, orders guided by ACF/PACF and information criteria |
| F4 | Simple forecast combination of the best models (only if the tutorials/lectures cover it) |

Rules for this phase:
- Log each model to `results_log.md`: specification, hyperparameters/orders, validation MSE (4 d.p.),
  MSE by horizon if useful, and a one-line comment.
- Write one forecasting function per model that takes a training series and a horizon and returns the forecasts,
  so the same code is used in validation and in the final fit.
- If you transform the series (log / Box-Cox), evaluate MSE on the original CPI scale.
- Check residual diagnostics for the main candidates.

**Checkpoint 2:** show me the comparison table and recommend a final model. Base the recommendation on validation MSE;
when models are close, also weigh simplicity, stability across origins and interpretability. Wait for my choice.

### Phase 3 — Final model and deliverables

1. Refit the chosen model on all 100 training quarters and forecast h = 1, …, 8.
2. Plot the training data with the forecasts (and intervals, if the tutorials show them); show residual diagnostics.
3. Write `CPI_forecast.csv`:
   - exactly two columns, `Quarter` then `CPI`;
   - 8 rows, `Mar-25`, `Jun-25`, …, `Dec-26` (same quarter format as `CPI_train.csv`);
   - add `assert` checks for all of the above.
4. Put this final cell at the end of the notebook, and **do not execute it**:
   ```python
   import pandas as pd

   CPI_test = pd.read_csv("CPI_test.csv")

   # YOUR CODE HERE : code that produces the test MSE test_error
   y_te = CPI_test["CPI"].values
   test_error = np.mean((final_forecast - y_te) ** 2)

   print(test_error)
   ```
   `final_forecast` is the 8-step forecast array created earlier. The cell must stay self-contained using objects
   created earlier, and must compute the same MSE as the formula in the spec.
5. Test the final cell: create a temporary fake `CPI_test.csv` (same columns, 8 rows) in a scratch copy, run the cell,
   then delete the fake file. Never leave it in the folder (it is in `.gitignore`).
6. Restart the kernel and run all cells except the last one (e.g. `jupyter nbconvert --execute` on a copy without
   the last cell). Every cell must run without errors.

**Checkpoint 3:** report the final validation MSE and confirm the notebook and CSV checks passed.

### Phase 4 — Write-up support

The notebook itself is assessed (14 marks), so the markdown must explain the procedure, but **I write the
interpretation and discussion myself**. Create `report_notes.md` with the key numbers (4 d.p.), the reasoning chain
for each modelling decision, and a list of limitations, to help me write those parts. Do not write my discussion prose.

---

## 3. Code style rules (strict)

1. **Tutorial first.** For every method, reuse the tutorial's functions, argument patterns and naming.
2. **Allowed methods only:** those that appear in my lectures/tutorials (list confirmed in Phase 0).
   No machine-learning forecasters (trees, boosting, neural nets, Prophet, auto-ML) unless I approve.
3. **Mark every deviation.** If a method has no tutorial code, or you must deviate, add a comment
   `# NOTE: not from tutorial — reason: ...` and list it in `tutorial_patterns.md` under Gaps.
4. **Readable over clever.** One idea per cell; each code cell preceded by a markdown cell explaining *what* and *why*
   (English, concise); comments on non-obvious lines; no heavy abstractions or long one-liners.
5. **Reproducibility.** Fixed seeds (`np.random.seed(0)`), relative paths only, no hard-coded numbers copied from
   earlier outputs. The notebook must run from a folder containing only `CPI_train.csv` (plus `CPI_test.csv` for
   the marker).
6. Print numerical results rounded to 4 decimal places.

---

## 4. How to talk to me

- I am learning. Give **plain-language intuition first, then the formula/code**.
- Use **bilingual Chinese–English explanations in chat**, e.g. "季节性 (seasonality)". Keep all notebook content
  in English.
- Show concrete numbers from our data when explaining a concept.
- Keep chat summaries short and structured; point me to files/cells instead of dumping raw output.
- If something in the assignment or the tutorials is ambiguous, ask me rather than guessing.

---

## 5. AI-use record (required by the assignment)

The spec requires a record of AI use and copies of AI outputs. Maintain `ai_use_log.md` in this folder.
For each session, add the date, what I asked, what you produced (files/cells), and which decisions I made myself.

At the end, draft a short AI-use statement for the end of my notebook that names the tool (Claude Code / Claude
model version), the publisher (Anthropic), and how it was used. I will review and edit it.

The spec says the notebook must not be generated entirely by AI and must show my own understanding and decisions.
So at each checkpoint, clearly separate what you propose from what I decide, and leave the interpretation of
results to me.
