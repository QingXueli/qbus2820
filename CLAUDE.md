# CLAUDE.md — Instructions for QBUS2820 Assignment 1

You are helping me (Sherry, a university student) complete QBUS2820 Assignment 1: predicting `WeeklyRent`.
Read `QBUS2820_A1_context.md` first. It contains the assignment specification, marking criteria, a data profile,
lecture knowledge from Weeks 1–7, and baseline results.

**The most important rule:** the code must look like the code I learned in my tutorials.
My tutorial notebooks are the style guide and the method whitelist. Imitate them; do not invent your own style.

---

## 1. Folder layout (expected)

```
project/
├── CLAUDE.md                        ← this file
├── QBUS2820_A1_context.md           ← assignment + data + lecture summary
├── WeeklyRent_training.csv
├── WeeklyRent_test_noLabel.csv
├── tutorials/                       ← my tutorial notebooks / .py files / solutions (Weeks 1–7)
└── (outputs you create)
    ├── tutorial_patterns.md
    ├── SID_Assignment1_implementation.ipynb
    ├── SID_Assignment1_prediction.csv
    ├── figures/
    ├── results_log.md
    └── ai_use_log.md
```

If `tutorials/` is missing or empty, **stop and ask me to add the files**. Do not continue on generic code.

---

## 2. Work in phases. Stop at each checkpoint and wait for my approval.

### Phase 0 — Learn my tutorial code (no modelling yet)

Read every file in `tutorials/`. Then write `tutorial_patterns.md` with these sections:

1. **Libraries and imports.** List exactly which libraries appear (e.g. `pandas`, `numpy`, `matplotlib`, `seaborn`, `statsmodels`, `sklearn`), plus the import aliases the tutorials use.
2. **Method → tutorial code map.** For each method, record:
   - which tutorial file and cell it appears in;
   - the exact functions and classes used;
   - a short code excerpt.

   The methods to map are: OLS (sklearn and/or statsmodels formula), dummies, interactions, polynomial terms, KNN, train/validation split, K-fold CV, AIC/BIC, best subset / forward / backward selection, ridge, lasso, elastic net, standardisation.
3. **Coding style.** Record:
   - variable naming (e.g. `X_train`, `y_train`, `df`);
   - whether the tutorials use `Pipeline` or manual steps;
   - how they compute MSE;
   - how they plot (figure size, titles, labels);
   - how they write markdown explanations and comments;
   - loop style for tuning (manual `for` loops vs `GridSearchCV`).
4. **Gaps.** List methods from the lectures that have no tutorial code.

**Checkpoint 0:** show me `tutorial_patterns.md` and a short summary. Wait for my OK.

### Phase 1 — Data understanding and EDA

Put this in the notebook, in tutorial style:
- load the data;
- check shape, dtypes, missing values and duplicates;
- produce summary statistics (4 decimal places);
- compare train and test covariate distributions;
- check whether `DistanceCBD = 40` is a cap, and look at the `PropertyAge` tail.

Plots to make:
- rent histogram;
- scatter (+ trend) of rent vs each continuous covariate;
- boxplots of rent by each binary variable and by `Bedrooms`;
- correlation heatmap;
- interaction exploration (e.g. rent vs distance, coloured by `HighDemandArea` / `NearTrain`);
- residual plots from a baseline OLS.

Save every figure to `figures/` with an informative title, axis labels and legend.

**Checkpoint 1:** summarise the EDA findings in 5–10 bullet points and propose candidate features, each with its justification. Wait for my OK.

### Phase 2 — Modelling

Use one fixed CV scheme for every model: the same `KFold` with `shuffle=True` and a fixed `random_state`, or whatever split the tutorials used. Compare the following models.

| ID | Model |
|---|---|
| M0 | Null model (training mean) |
| M1 | OLS with all 7 linear terms |
| M2 | OLS with EDA/domain-driven nonlinear terms and interactions (hierarchy principle) |
| M3 | Subset selection over the expanded features (forward/backward stepwise with AIC/BIC/CV) |
| M4 | Ridge / Lasso / Elastic Net on an expanded (e.g. polynomial) basis, standardised, λ chosen by CV (and AIC/BIC if the tutorials do that) |
| M5 | KNN with scaled predictors, k chosen by CV |

Rules for this phase:
- Log each model to `results_log.md`: features, hyperparameters, CV MSE (4 d.p.), its standard error, and a one-line comment.
- Use one `make_features(df)` function for both train and test, so the final cell can reuse it.
- Fit scalers and any preprocessing on training data only (inside CV folds when tuning).
- If you log-transform the response, evaluate MSE on the original AUD scale.

**Checkpoint 2:** show me the comparison table and recommend a final model. Base the recommendation on CV MSE; when the gap is within about 1 SE, also weigh simplicity and interpretability. Wait for my choice.

### Phase 3 — Final model and deliverables

1. Refit the chosen model on all 5000 training rows.
2. Show residual diagnostics and a coefficient table with business interpretation (e.g. "holding other factors fixed, near train ≈ +$X/week").
3. Write `SID_Assignment1_prediction.csv`:
   - exactly one column, named `WeeklyRent`;
   - 1000 rows, in the same order as `WeeklyRent_test_noLabel.csv`;
   - add `assert` checks for all of the above.
4. Put this exact final cell at the end of the notebook, and **do not execute it**:
   ```python
   import pandas as pd
   WeeklyRent_test = pd.read_csv("WeeklyRent_test.csv")
   # YOUR CODE HERE : code that produces the test MSE test_error
   X_te = make_features(WeeklyRent_test)
   y_te = WeeklyRent_test["WeeklyRent"]
   test_error = np.mean((y_te - final_model.predict(X_te)) ** 2)
   print(test_error)
   ```
   Adapt it if the final model needs a back-transform, but it must stay self-contained using objects created earlier.
5. Test that the final cell works:
   - create a temporary fake `WeeklyRent_test.csv` from a held-out slice of training data;
   - run the final cell in a scratch copy;
   - delete the fake file afterwards;
   - never leave it in the folder.
6. Restart the kernel and run all cells except the last one (e.g. `jupyter nbconvert --execute` on a copy without the last cell). Every cell must run without errors.

**Checkpoint 3:** report the final CV MSE and confirm the notebook and CSV checks passed.

### Phase 4 — Report support (I write the report myself)

Create `report_notes.md` containing:
- the key numbers (4 d.p.);
- which figures and tables to use, with suggested captions;
- the reasoning chain for each modelling decision;
- a list of limitations.

Do **not** write the full report prose.

---

## 3. Code style rules (strict)

1. **Tutorial first.** For every method, reuse the tutorial's functions, argument patterns and naming. Where the tutorial uses a manual loop, use a manual loop. Where it uses statsmodels formulas, use statsmodels formulas.
2. **Allowed methods only:** those in the context file §4 (Weeks 1–7). No trees, forests, boosting, SVR, neural nets or GAM libraries.
3. **Mark every deviation.** If a method has no tutorial code, or you must deviate, add a comment `# NOTE: not from tutorial — reason: ...` and list it in `tutorial_patterns.md` under Gaps. Keep deviations minimal and simple.
4. **Readable over clever.**
   - One idea per cell.
   - Each code cell is preceded by a markdown cell explaining *what* and *why* (English, concise).
   - Use comments on non-obvious lines.
   - Avoid heavy abstractions, classes and long one-liners.
5. **Reproducibility.**
   - Fixed seeds.
   - Relative paths only.
   - No hard-coded numbers copied from earlier outputs.
6. Print numerical results rounded to 4 decimal places.

---

## 4. How to talk to me

- I am learning. When you explain a step, give **plain-language intuition first, then the formula/code**.
- Use **bilingual Chinese–English explanations in chat**, e.g. "交叉验证 (cross-validation)". Keep all notebook content in English.
- Show concrete numbers from our data when explaining a concept.
- Keep chat summaries short and structured. Don't dump raw output; point me to the file or cell instead.
- If something in the assignment or the tutorials is ambiguous, ask me rather than guessing.

---

## 5. AI-use record (required by the assignment)

Maintain `ai_use_log.md`. For each session, add:
- the date;
- what I asked;
- what you produced (files/cells);
- which decisions I made myself.

At the end, draft a short AI-use statement for my report that names the tool (Claude Code / Claude model version), the publisher (Anthropic), and how it was used. I will review and edit it.
