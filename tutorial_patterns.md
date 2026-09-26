# Tutorial Patterns (Phase 0) — PARTIAL

> Status: **only Week 1 received** (`tutorials/week01/`). Weeks 2–7 are still needed before the
> method map below can be filled in. `QBUS2820_A1_context.md` is also still missing.

Files read:

| File | Cells | Content |
|---|---|---|
| `week01/Tutorial_01_task_1_solution.ipynb` | 3 code, 0 markdown | Load Excel, plot average spend by year (two alternatives) |
| `week01/Tutorial_01_task_2_solution.ipynb` | 5 code, 0 markdown | Load Excel, histograms (all / subset / two groups overlaid), group mean & variance |

---

## 1. Libraries and imports

Exactly these, in this order, at the top of the first cell (no separate import cell):

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
```

- Data is loaded in the **same cell** as the imports: `df = pd.read_excel("DirectMarketing.xlsx")`, then `.head()`.
- Not seen yet: `seaborn`, `statsmodels`, `sklearn`.

## 2. Method → tutorial code map

| Method | Tutorial / cell | Functions & classes | Excerpt |
|---|---|---|---|
| Load data | T1 task 1 cell 0; task 2 cell 0 | `pd.read_excel` | `df = pd.read_excel("DirectMarketing.xlsx")` |
| Select column | T1 task 2 cell 0 | `df['col']` | `salary = df['Salary']` |
| Filter rows (boolean mask) | T1 task 2 cells 2–3 | `df[df['col'] > v]` | `good_spenders = df[ df['AmountSpent'] > 500]` |
| Drop index column | T1 task 1 cell 2 | `.iloc[:, 1:]` | `spend_years = spend_years.iloc[:,1:]` |
| Group mean / variance | T1 task 2 cell 4 | `.mean()`, `.var()` | `avg_male_spent = male_cust_spent['AmountSpent'].mean()` |
| Histogram | T1 task 2 cells 1–3 | `plt.hist` | `plt.hist(salary)` |
| Overlaid histograms by group | T1 task 2 cell 3 | `plt.hist(..., alpha=0.5, label=...)`, `plt.legend()` | see below |
| Line plot | T1 task 1 cells 1–2 | `plt.plot`, `plt.xticks` | `plt.plot(spend_years.mean()[1:6])` |
| OLS, dummies, interactions, polynomial terms, KNN, train/validation split, K-fold CV, AIC/BIC, subset/forward/backward selection, ridge, lasso, elastic net, standardisation | — | — | **Not yet available (Weeks 2–7 needed)** |

## 3. Coding style

- **Variable naming:** `df` for the main DataFrame; descriptive snake_case for subsets and statistics
  (`good_spenders`, `male_cust_spent`, `avg_male_spent`, `var_male_spent`).
- **Pipeline vs manual:** manual, step-by-step code; no functions or classes defined.
- **MSE:** not seen yet.
- **Plotting pattern** (matplotlib only, default figure size, one figure per cell):
  ```python
  fig = plt.figure()

  plt.hist(male_cust_spent['AmountSpent'], alpha = 0.5, label = "Male")
  plt.hist(female_cust_spent['AmountSpent'], alpha = 0.5, label = "Female")

  plt.legend()
  plt.xlabel("Amount Spent")
  plt.ylabel("Count")
  plt.title("Spending Distributions of Male and Female Cohorts")
  plt.savefig("part3.jpg")

  plt.show()
  ```
  Always sets `xlabel`, `ylabel`, `title`; saves with `plt.savefig(...)` before `plt.show()`.
- **Printing numbers:** `str.format` style, e.g.
  `print("MALE - ${0:.2f} mean spending with variance {1:.2f}".format(avg_male_spent, var_male_spent))`
  (the assignment needs 4 d.p., so we will use `{0:.4f}`).
- **Markdown / comments:** solutions contain no markdown cells and almost no comments (only `# Alternative 1`).
  CLAUDE.md requires a markdown cell before each code cell, so that will be added on top of this style.
- **Tuning loops:** not seen yet.

## 4. Gaps

Everything modelling-related currently has no tutorial code, because only Week 1 has been provided.
Lecture methods cannot be listed here until `QBUS2820_A1_context.md` (§4) is available.
