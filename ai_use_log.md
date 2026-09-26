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
