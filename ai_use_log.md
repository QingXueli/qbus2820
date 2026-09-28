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

## 2026-09-27 (Phase 2 — restructure)
**What I asked**: restructure the modelling as "build candidate models first (M0, best subset over 128 subsets of the
7 predictors, stepwise, ...), then validate all with CV and choose"; forward and backward stepwise each with AIC and BIC;
no standardisation using all data (leakage).
**What the AI produced**: rewrote Section 3: 3.1 set-up (`make_features`, `kf`), 3.2 model building (M0, M1, best subset
M2, forward stepwise M3, backward stepwise M4, ridge/lasso/elastic net M5 and KNN M6 defined with Pipelines),
3.3 validation (same 10 folds; best-subset size, λ, l1_ratio and k chosen by manual CV loops), 3.4 comparison table and
chart; updated `results_log.md` and `tutorial_patterns.md`.
**Decisions I made myself**: two-stage structure; AIC and BIC versions of both stepwise methods; scaling inside folds only.
- Asked what 3.1 Set-up does and why the baseline models had no code. Decision (mine): start Section 3 with the baseline
  models. The AI reordered Section 3: 3.1 baseline models (M0 mean, M1 OLS with training MSE, R², AIC, BIC),
  3.2 best subset, 3.3 expanded features (`make_features` introduced here) and stepwise/regularised/KNN models,
  3.4 validation (`KFold` defined here), 3.5 comparison. Results unchanged.
- Asked how the tutorials organise model building and CV (answer: CV is computed right after each model is built or
  while tuning it; models are compared with the same KFold in one table, W6). Decision (mine): restructure Section 3 in
  that style. The AI rewrote Section 3 as 3.1 CV set-up, 3.2 M0/M1, 3.3 best subset, 3.4 expanded features,
  3.5 forward, 3.6 backward, 3.7 ridge/lasso/elastic net, 3.8 KNN (each evaluated by CV immediately), 3.9 comparison.
  Results unchanged.
- Decision (mine): remove the elastic net (not covered in my tutorials; it did not beat the lasso). Section 3.7 now has ridge and lasso only.
- Decision (mine): add the elastic net back (it is in Lectures 6–7), and use the EDA interaction findings explicitly in a
  model. The AI restored the elastic net and added "3.5 M3: EDA-driven OLS" (squared terms from 2.6/2.7 and interactions
  from 2.9, with a table linking each term to its EDA evidence, p-values and CV MSE 2075.50); later models renumbered
  (M4 forward, M5 backward, M6 ridge/lasso/elastic net, M7 KNN; Section 3.10 comparison).
- Decision (mine): keep M3 (EDA-driven OLS) — it tests the EDA hypotheses and links the EDA to the modelling.
- Asked to understand every line of Section 3; the AI wrote notes/Section3_code_explained.md (line-by-line, with English comment versions).
- Decision (mine): notebook markdown should be short (what/why in 1–3 sentences); details go to the report.
  The AI shortened all markdown cells (≈2300 → ≈980 words), added a short result after the model comparison
  ("recommended", final choice pending), and created `report_notes.md` with the removed details, key numbers,
  reasoning chain, suggested figures/captions and limitations.
- Sherry found the one-line M1 cell (`cv_mse(...)`) hard to follow. At her request, M1 was rewritten step by step in Tutorial 6 style (create `LinearRegression` object → fit and print coefficients → `cross_val_score` on `kf` → fold MSEs → mean and SE). `cv_mse` was moved to just after M1 and is now introduced as a wrapper around those same steps. CV results are unchanged (M1 3302.3821, SE 29.6067).
- Decision (mine): I noticed the expanded M1 CV cell duplicated `cv_mse`. `cv_mse` was moved back into 3.1 (with line comments), and M1 now calls it; the M1 fit/coefficient cell keeps its step-by-step form with English comments. Results unchanged.
- M2 (best subset): at my request the cell now prints the M2 CV MSE and SE right after the model; added line comments (inner loop = keep the smallest RSS of each size), corrected the markdown to 127 non-empty subsets (128 incl. M0), and extended the result note (best 1-predictor model not nested in best 2-predictor model).
- Decision (mine): after reviewing what M3 does (EDA-driven terms; bridge between EDA and stepwise selection), I decided to keep M3.
- M4 (forward stepwise): added the missing `# NOTE: not from tutorial` line and line comments; results unchanged.
- Decision (mine): AIC/BIC in M4/M5 now computed with the Tutorial 6 formula (n*log(RSS/n) + penalty*d, d = intercept + slopes + error variance) via a small `aic_bic` function, instead of statsmodels `est.aic`/`est.bic`. Values differ by a constant only; selected terms and CV MSEs unchanged. Added NOTE line and comments to M5.
- Decision (mine): shortened the AIC/BIC function to the Tutorial 6 formula using sigma2_hat = est.ssr / n (no extra sklearn refit). Correction: est.aic/est.bic do not appear in the tutorials (Tutorial 7 only reads AIC/BIC from est.summary()), so the Tutorial 6 formula is kept. Results unchanged.
- M4 output: at my request the result cell now prints a small AIC/BIC table (extra terms, CV MSE, SE) and a numbered list of the added terms (marking the AIC-only term), plus a one-line Result note. Numbers unchanged.
- M5 (backward stepwise): at my request, markdown rewritten in plain language (start from the biggest model, drop the least useful term one at a time); code split into path cell + result cell like M4; result prints a small AIC/BIC table, the removal order (marking the BIC-only removal) and the kept terms; result note now explains the AIC/BIC plot. Numbers unchanged.
- AIC/BIC path plot: added dashed vertical lines at the AIC and BIC minima (labelled 'AIC min = 7', 'BIC min = 6'), at my request.
- Decision (mine): no three- or four-way interactions. The AI checked all 35 three-way terms on top of the BIC model (none lowered the CV MSE; best +0.13); result recorded in report_notes.md, not added to the notebook.
- M6: added line comments, a markdown with the ridge (L2) and lasso (L1) formulas, and a small cell (Tutorial 7 style) that refits ridge and lasso on all training data and counts zero coefficients (ridge 0/32, lasso 9/32, list of dropped terms). CV results unchanged.
- M6 made faster at my request (my local run was very slow): 20-point alpha grid and a single elastic net l1_ratio = 0.5, computed in the same loop as ridge and lasso (3-panel plot). M6 now runs in about 37 s instead of about 88 s on the server. New results: lasso 2012.2483 (alpha 0.0695), ridge 2017.8797 (alpha 0.4281), elastic net 2019.0313 (alpha 0.001, grid minimum). Final recommendation unchanged; notes and results log updated.
- M6 made lighter again because my laptop overheated: 10 alphas from 0.01 to 10 (np.logspace(-2, 1, 10)), elastic net l1_ratio = 0.9 (with 0.5 and alpha >= 0.01 the L2 part over-shrinks, CV MSE 2090). M6 now runs in about 13 s on the server. Results: lasso 2012.6979 (alpha 0.1, 12 zero coefficients), ridge 2017.8785 (alpha 0.4642), elastic net 2022.4385 (alpha 0.01, grid minimum). Recommendation unchanged.
- Decision (mine): M6 ridge and lasso now choose λ by AIC(λ)/BIC(λ) with effective degrees of freedom (Lectures 6–7; Tutorial 7 homework) over 30 values 0.001–100, fitted once per λ on data standardised with training mean/std (Tutorial 7); the chosen models are then compared by the same 10-fold CV (pipeline). Elastic net keeps a small CV grid (no df formula in the lectures). No relaxed lasso. M6 runs in about 11 s. Recommendation unchanged.
- M6: zero-coefficient output turned into two tables at my request (summary of zero/non-zero counts; 32-row ridge vs lasso coefficient table with a 'Dropped by lasso' column).
- M6 coefficient output shortened at my request: summary table + only the 12 terms the lasso drops (ridge vs lasso coefficient).
- M6: fuller English line comments added to all five cells at my request (results unchanged).
- Decision (mine): no bias–variance plot for ridge; the training vs CV MSE numbers are kept in report_notes.md for the report.
- Model comparison plot redesigned at my request: dot-and-whisker (CV MSE ± 1 SE) instead of bars cut at 1900, coloured by model family, value labels, shaded one-standard-error band; result note corrected (M3 is outside the 1-SE band).

### Phase 3 (final model)
- Decision (mine): final model = M4 (forward stepwise, BIC).
- The AI added Section 4 to the notebook: refit on all 5000 rows (statsmodels table + scikit-learn `final_model`),
  effects for an average property, residual diagnostics, the prediction CSV with assert checks, and the marker-only final
  cell (appended unexecuted). It tested the final cell in a scratch copy with a fake `WeeklyRent_test.csv` built from training
  rows 4000–4999 (output matched an independent calculation), deleted the fake file, and re-ran all other cells without errors.

### Phase 4 (report support)
- Sherry uploaded her own notebook, the assignment spec and the marking criteria. The AI checked the notebook: all cells run
  from a clean kernel; the marker-only final cell was commented out and executed, with an empty cell after it (to be fixed by
  Sherry). With the final cell uncommented, a scratch run with a fake test file gave test_error 2034.7045 (same as the repo notebook).
- The AI added a report outline mapped to the marking criteria and a draft AI-use statement to report_notes.md. Sherry writes the report.
- Sherry asked whether combining models (e.g. polynomial + lasso) could lower the CV MSE. The AI tested cubic terms, cap indicators, degree-3 polynomial + lasso, averaging OLS and lasso predictions, and a log response (scratch only); none improved the BIC model beyond noise. Recorded in report_notes.md.
- Report: Sherry wrote a Chinese draft of the executive summary; the AI gave feedback and, at her request, revised her draft
  (added MSE definition, feature list, EDA link, null and EDA-driven models, 1-SE selection rule, SE and % vs null model,
  replaced the unsupported ranking of drivers with dollar effects, added interaction numbers and an optional application/
  limitation paragraph). Sherry will translate it into English herself.
