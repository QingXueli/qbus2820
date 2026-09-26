# AI Use Log — QBUS2820 Assignment 1

Tool: Claude Code (Anthropic), running in a Claude Code on the web session.

---

## 2026-09-23

**What I asked**
- Break the assignment into steps and give a plan.
- Upload the spec and data to the repo and scaffold a notebook (EDA, CV comparison, final-cell template).

**What the AI produced**
- A step-by-step plan (in chat), including a first look at the data and baseline 5-fold CV MSEs for a few models.
- Repo files: `spec/QBUS2820_S2_2026_Assignment1.pdf`, the two CSVs, `README.md`, `.gitignore`,
  `SID_Assignment1_implementation.ipynb` (scaffold) and `SID_Assignment1_prediction.csv`.
- Verified the marker-only final cell with a fake `WeeklyRent_test.csv` in a scratch copy (file not kept).

**Decisions I made myself**
- Asked for the scaffold to be placed on branch `claude/project-breakdown-plan-5kn712`.

---

## 2026-09-26

**What I asked**
- A line-by-line explanation of the Setup cells (imports, seed, data loading).
- Provided `CLAUDE.md` with working rules (tutorial-style code, phased checkpoints, allowed methods Weeks 1–7).

**What the AI produced**
- Explanation in chat of every import and setup line.
- Saved `CLAUDE.md` to the repo; created this log and an empty `tutorials/` folder.
- Noted that the existing scaffold uses methods outside the allowed list (trees, random forest, boosting, splines)
  and `Pipeline`/`GridSearchCV`, which must be revised to match tutorial code after Phase 0.

**Decisions I made myself**
- Set the working rules in `CLAUDE.md` (tutorial code as style guide and method whitelist; checkpoints).

**Later on 2026-09-26**
- I uploaded Week 1 tutorial solutions and data; the AI saved them to `tutorials/week01/` and wrote a partial
  `tutorial_patterns.md` (Week 1 style only). Weeks 2–7 and `QBUS2820_A1_context.md` still to be provided.
- I uploaded Week 2 tutorial files; the AI saved them to `tutorials/week02/` and updated `tutorial_patterns.md`
  (statsmodels OLS / formula API, dummies, correlation, pairplot). Weeks 3–7 still to be provided.
- I uploaded Week 3 tutorial files; the AI saved them to `tutorials/week03/` and updated `tutorial_patterns.md`
  (train/test split with `df.sample`, KNN with a manual `for` loop over k, `mean_squared_error`). Weeks 4–7 still to be provided.
- I uploaded Week 4–5 tutorial files; the AI saved them to `tutorials/week04/` and `tutorials/week05/` and rewrote
  `tutorial_patterns.md` (KFold, `cross_val_score` with `neg_mean_squared_error`, CV loop for k, KNN with Mahalanobis
  distance, `knn_test` helper). Weeks 6–7 still to be provided.
- I uploaded Week 6–7 tutorial files; the AI saved them to `tutorials/week06/` and `tutorials/week07/` and completed
  `tutorial_patterns.md` (GridSearchCV, PolynomialFeatures + CV, AIC/BIC, backward elimination, manual standardisation,
  Lasso/LassoCV) with a Gaps table. Checkpoint 0 reached; waiting for my approval.

## 2026-09-26 (Phase 1 — EDA)

**What I asked**
- Decisions at Checkpoint 0: use a `Pipeline` so standardisation happens inside each CV fold; asked how to choose interactions;
  asked to do the EDA first (variable types, missing values, outliers, response distribution, correlations; histograms, scatter plots, heatmap).

**What the AI produced**
- Rebuilt `SID_Assignment1_implementation.ipynb` in tutorial style with Setup + EDA (Sections 1–2):
  structure, missing/duplicates, summary statistics (4 d.p.), train vs test comparison, boundary values (DistanceCBD = 40,
  FloorArea = 35), PropertyAge tail, response histogram, scatter plots with linear/quadratic fits, box plots, correlation
  table and heatmap, interaction plots, one-term-at-a-time interaction screening (OLS + AIC), baseline OLS residual plots,
  and a summary of findings. Figures saved to `figures/fig01`–`fig09`.
- Removed the old modelling scaffold (non-tutorial methods) and the stale prediction CSV; updated `tutorial_patterns.md` and `README.md`.

**Decisions I made myself**
- Standardisation inside CV folds via `Pipeline`.
- Do EDA first; modelling decisions deferred.
- Asked for a line-by-line explanation of the EDA code and whether the setup lines are needed / from the tutorials.
  The AI wrote `notes/EDA_code_explained.md` and tested that removing the warnings/seaborn-style lines produces no warnings or errors.
- Decision (mine): keep the tutorial setup lines (warnings, `%matplotlib inline`, `sns.set_context`, `sns.set_style`);
  do not save figures to files — plots are only shown in the notebook and I will take screenshots for the report.
  The AI removed `os.makedirs` and all `plt.savefig` calls, deleted `figures/`, and re-ran the notebook (no errors).
- Asked what the summary statistics, missing-value checks and the train-vs-test histograms mean, and for tips on writing
  English code comments; the AI explained in chat (no file changes).
- Decision (mine): no extra `# NOTE` comments for `isna()`, `duplicated()`, `describe()` (not in tutorials, but basic checks
  required by CLAUDE.md Phase 1).
- Decision (mine): keep the train-vs-test comparison (table and histograms) in Section 2.4, although it is not in the
  tutorials or the assignment spec, because it supports using CV error as a guide to test error.
- Decision (mine): keep Section 2.5 in a shorter form — the two histograms were removed; the boundary-value counts and the
  PropertyAge quantiles remain, followed by a short conclusion. The AI also removed an unsupported sentence from the
  EDA summary (2.12) about residuals at the capped values, since that check was not in the notebook.
- Decision (mine): delete Section 2.5 (boundary values / PropertyAge tail) and instead check outliers with standardised
  residuals from the baseline OLS. The AI removed 2.5, renumbered later sections (2.5–2.11), added the outlier check
  (17 observations with |z| > 3 vs 13.5 expected; mainly distant or five-bedroom properties; all kept) and rewrote
  summary item 3.
- Asked what Section 2.6 does and for English comment tips; decision (mine): show the three scatter plots in one row.
  The AI changed Section 2.6 to a 1 x 3 `plt.subplots` layout (comments added) and re-ran the notebook (no errors).
- Decision (mine): quadratic-fit line in Section 2.6 changed from orange to green for contrast.
- Decision (mine): write polynomial fits in separate steps (transformer -> fit_transform -> LinearRegression() -> fit), as in Week 6; applied in Sections 2.6 and 2.10.
- Decision (mine): in Section 2.7, replace the list comprehension with an explicit for loop + append (tutorial style).
- Asked for a summary of Section 2.7; the AI added a markdown summary cell after the group-means table.
- Asked to show the correlation of every predictor with WeeklyRent as a chart, and for the Section 2.8 conclusion.
  The AI added a sorted correlation table (Tutorial 5 style), a horizontal bar chart (marked NOTE) and a short summary.
  Also discussed multicollinearity: VIF not in tutorials; decided the correlation matrix (W5 approach) is sufficient for now.

## 2026-09-26 (lectures)
- I uploaded the Week 1–7 lecture slides. The AI saved them to `lectures/`, wrote `lecture_summary.md`
  (topics per lecture, allowed methods, key points for modelling) and noted in `tutorial_patterns.md` that the
  gap methods (interactions, subset selection, ridge, elastic net) are covered in the lectures.
- Asked the AI to do the interaction analysis (Section 2.9). The AI rewrote 2.9 following Lecture 2 ("domain knowledge,
  or trial and error") and Lectures 4–5 (AIC): commented group-slope plots, base OLS AIC, one-at-a-time AIC screening of
  all 21 pairwise interactions (plain nested loops instead of itertools) and of 4 squared terms, and a short summary.
- Decision (mine): renamed the Section 2.9 figure title (was 'Separate Linear Fits by Group'); the AI also added a title to each panel.
- Decision (mine): AIC screening of candidate terms is a modelling step, so it was moved out of the EDA (option A).
  Section 2.9 now contains only the group-slope plots and their interpretation; summary item 8 was rewritten from the
  plots. The screening code and its results are kept in `notes/phase2_candidate_screening.md` for the modelling section.
- Decision (mine): remove the residual plots from Section 2.10 (not in tutorials or lectures; curvature already shown in 2.6),
  overriding the CLAUDE.md Phase 1 item "residual plots from a baseline OLS". 2.10 is now "Baseline OLS model and outlier
  check" (OLS summary + standardised-residual outlier check). Summary items 5 and 9 no longer refer to residual plots.
  The AI also corrected an earlier chat statement: sigma_hat of the baseline OLS is 57.4 AUD, not 43.
- Asked how to structure 2.10; the AI added comments to the baseline OLS cell and a short interpretation of its summary before the outlier check.

## 2026-09-26 (Phase 2 — modelling)
**What I asked**: start the model selection part: a base model, then the models from the tutorials, then choose the best.
**What the AI produced**: notebook Section 3 (3.1–3.9): 10-fold CV set-up and helpers, `make_features(df)` (32 columns),
M0 null model, M1 OLS, M2 AIC-screened OLS, M3 backward selection (AIC and BIC), M4 ridge/lasso/elastic net in a
Pipeline, M5 KNN with k chosen by CV, comparison table and bar chart; `results_log.md`. The AI checked that data-driven
term selection was optimistic when done on all data and made M2/M3 selection nested inside the CV folds.
**Decisions I made myself**: overall approach (base model, tutorial models, pick the best). Final model choice pending (Checkpoint 2).
