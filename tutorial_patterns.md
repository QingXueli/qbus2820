# Tutorial Patterns (Phase 0) — PARTIAL

> Status: **Weeks 1–2 received.** Weeks 3–7 and `QBUS2820_A1_context.md` are still needed
> before the method map can be completed.

Files read:

| File | Cells | Content |
|---|---|---|
| `week01/Tutorial_01_task_1_solution.ipynb` | 3 code | Load Excel, line plot of average spend by year |
| `week01/Tutorial_01_task_2_solution.ipynb` | 5 code | Histograms (all / subset / two groups overlaid), group mean & variance |
| `week02/Tutorial_02.ipynb` | 61 (markdown + code) | Dummies, correlation, scatter, SLR/MLR with `statsmodels`, matrix OLS, formula API, plotting the fit, prediction |
| `week02/Tutorial_02_task.ipynb` | 1 markdown, 1 empty code | Task: OLS of `AmountSpent` on Salary, Children, Gender, Catalogs; significance, R², prediction |

---

## 1. Libraries and imports

| Library | Alias | First seen | Used for |
|---|---|---|---|
| `pandas` | `pd` | W1 | `read_excel`, `get_dummies`, `.corr()`, DataFrames |
| `matplotlib.pyplot` | `plt` | W1 | all plots |
| `numpy` | `np` | W1 | `linspace`, `column_stack`, `linalg.inv`, `min`/`max` |
| `statsmodels.api` | `sm` | W2 | `sm.add_constant`, `sm.OLS` |
| `statsmodels.formula.api` | `smf` | W2 | `smf.ols(formula=..., data=...)` |
| `seaborn` | `sns` | W2 | `sns.pairplot` only |

- Week 1: all imports in the first cell together with loading data.
- Week 2: imports appear **where first needed** (e.g. `import statsmodels.api as sm` in the cell that first uses it).
  For the assignment notebook we will keep them together at the top, which is clearer for the marker.
- `sklearn`: not seen yet.

## 2. Method → tutorial code map

| Method | Tutorial / cell | Functions & classes | Excerpt |
|---|---|---|---|
| Load data | W1 T2 c0; W2 c1 | `pd.read_excel` | `br = pd.read_excel("BatonRouge.xls")` |
| Filter rows | W1 T2 c2–3 | boolean mask | `good_spenders = df[ df['AmountSpent'] > 500]` |
| Group mean / variance | W1 T2 c4 | `.mean()`, `.var()` | `male_cust_spent['AmountSpent'].mean()` |
| Correlation matrix | W2 c9 | `df.corr()` | `correlations = br.corr()` |
| Pair plot | W2 c41 | `sns.pairplot` | `sns.pairplot(br)` |
| **Dummies** | W2 c5–7 | `pd.get_dummies(..., drop_first=True)` | `s = pd.get_dummies(s, drop_first=True)` |
| **OLS (statsmodels, matrix)** | W2 c13–15, c38–42 | `sm.add_constant(x, prepend=True)`, `sm.OLS(y, X).fit()`, `.summary()` | see below |
| **OLS (statsmodels formula)** | W2 c28 | `smf.ols(formula=..., data=...)` | `smf.ols(formula = "Price ~ SQFT", data = br)` |
| OLS by normal equations | W2 c18, c53 | `np.linalg.inv(X.T*X) * X.T * y` | teaching only; not needed for the assignment |
| Plot fitted line | W2 c30–34 | `np.linspace`, `results.params`, `results.predict` | `x_points = np.linspace(min_x, max_x, 100)` |
| Prediction | W2 c56–60 | `results.predict(list)`; formula: `results_formula.predict(pd.DataFrame([...]))` | `results_formula.predict( new_house_df )` |
| Interactions, polynomial terms, KNN, train/validation split, K-fold CV, AIC/BIC, subset/forward/backward selection, ridge, lasso, elastic net, standardisation, MSE | — | — | **Not yet available (Weeks 3–7 needed)** |

OLS pattern (W2):
```python
import statsmodels.api as sm

x = br.iloc[:, 1:]
x_with_intercept = sm.add_constant(x, prepend=True)

model_multivariate = sm.OLS(y, x_with_intercept)
results_multivariate = model_multivariate.fit()
print(results_multivariate.summary())
```

Formula pattern (W2):
```python
import statsmodels.formula.api as smf

formula = "Price ~ SQFT"
model_formula = smf.ols(formula = formula, data = br)
results_formula = model_formula.fit()
```
Formula strings also support interactions (`a:b`, `a*b`) and transforms (`I(x**2)`, `np.log(x)`);
whether the tutorials use these will be confirmed from later weeks.

## 3. Coding style

- **Variable naming:** main DataFrame named after the dataset (`br`) or `df`; `y` = target Series, `x` = features;
  `x_with_intercept`; `model_*` for the unfitted model and `results_*` for the fitted one
  (`model_multivariate` / `results_multivariate`, `model_formula` / `results_formula`).
- **Pipeline vs manual:** manual, step by step; no user-defined functions or classes.
- **MSE:** not seen yet (W2 evaluates with R² from `.summary()`).
- **Plots:** matplotlib; `fig = plt.figure()` → plot → `plt.xlabel` / `plt.ylabel` (/ `plt.title`) → (`plt.savefig`) → `plt.show()`.
  Scatter uses `alpha` (0.1–0.25) for overplotting; fitted line in `color = "red"`. Default figure size.
  seaborn only for `pairplot`.
- **Printing numbers:** `print("beta_1: {0:.2f}".format(lin_beta))` → we use `{0:.4f}` for 4 d.p.
- **Markdown / comments:** W2 has a markdown cell before most code cells: a short heading plus 1–3 sentences
  on *what* and *why*, sometimes with LaTeX formulas. Comments are short (`# Add a column of ones for beta_0`).
- **Tuning loops:** not seen yet.

## 4. Gaps (so far)

- All methods after OLS/dummies still lack tutorial code (Weeks 3–7 needed).
- Lecture method list pending `QBUS2820_A1_context.md` §4.
