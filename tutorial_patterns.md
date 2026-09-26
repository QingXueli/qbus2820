# Tutorial Patterns (Phase 0) — PARTIAL

> Status: **Weeks 1–5 received.** Weeks 6–7 and `QBUS2820_A1_context.md` are still needed
> before the method map can be completed.

Files read:

| File | Cells | Content |
|---|---|---|
| `week01/Tutorial_01_task_1_solution.ipynb` | 3 code | Load Excel, line plot of average spend by year |
| `week01/Tutorial_01_task_2_solution.ipynb` | 5 code | Histograms (all / subset / two groups overlaid), group mean & variance |
| `week02/Tutorial_02.ipynb` | 61 (markdown + code) | Dummies, correlation, scatter, SLR/MLR with `statsmodels`, matrix OLS, formula API, plotting the fit, prediction |
| `week02/Tutorial_02_task.ipynb` | 1 markdown, 1 empty code | Task: OLS of `AmountSpent` on Salary, Children, Gender, Catalogs |
| `week03/Tutorial_03_task_solution.ipynb` | 9 | Credit data: 70/30 train/test split, KNN, loop over k = 1..100, test RMSE vs k |
| `week04/Tutorial_04_task_solution.ipynb` | 10 | Simulated quadratic data; bias–variance decomposition of KNN at one point (k = 1 vs 50) |
| `week05/Tutorial_05.ipynb` | 43 | Credit data: dummies, train/test split, `KFold`, LOOCV, `cross_val_score`, CV loop over k, KNN with Mahalanobis distance, `knn_test` helper, adding predictors |

(`week04/BatonRouge.xls` and `week04/Credit.csv` are identical data files re-supplied; no extra notebook used them.)

---

## 1. Libraries and imports

| Library | Alias / import | First seen | Used for |
|---|---|---|---|
| `pandas` | `pd` | W1 | `read_excel`, `read_csv(..., index_col=...)`, `get_dummies`, `.corr()` |
| `matplotlib.pyplot` | `plt` | W1 | all plots |
| `numpy` | `np` | W1 | `arange`, `linspace`, `argmin`, `mean`, `sqrt`, `random.seed`, `random.normal` |
| `statsmodels.api` | `sm` | W2 | `sm.add_constant`, `sm.OLS` |
| `statsmodels.formula.api` | `smf` | W2 | `smf.ols(formula=..., data=...)` |
| `seaborn` | `sns` | W2 | `sns.pairplot` (W2); `sns.set_context('notebook')` (W5) |
| `sklearn.neighbors` | `from sklearn.neighbors import KNeighborsRegressor` (W3, W5); `from sklearn import neighbors` (W4) | W3 | KNN regression |
| `sklearn.metrics` | `mean_squared_error`, `make_scorer` | W3 / W5 | MSE; custom scorer |
| `sklearn.model_selection` | `KFold`, `LeaveOneOut`, `cross_val_score` | W5 | cross-validation |

- W5 states the convention explicitly: common imports (`pd`, `np`, `plt`, `sns`, `%matplotlib inline`) in the first cell;
  *"We will load new packages and functions in context to make clear what we are using them for."*
- W5 first cell:
  ```python
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  import seaborn as sns
  %matplotlib inline
  sns.set_context('notebook') # optimises picture for notebook viewing
  ```

## 2. Method → tutorial code map

| Method | Tutorial / cell | Functions & classes | Excerpt |
|---|---|---|---|
| Load data | W2 c1; W5 c3 | `pd.read_excel`, `pd.read_csv` | `data=pd.read_csv('credit.csv', index_col='Obs')` |
| Correlation with response | W2 c9; W5 c31 | `.corr()` | `train.corr().round(3)['Balance']` |
| Pair plot | W2 c41 | `sns.pairplot` | `sns.pairplot(br)` |
| **Dummies** | W2 c5–7; W5 c5 | `pd.get_dummies(..., drop_first=True)`; or `(col == value).astype(int)` | `data['Student']=(data['Student'] =='Yes').astype(int)` |
| Drop collinear predictor | W5 c5 | `.drop(col, axis=1)` | `data=data.drop('Rating', axis=1)` |
| **OLS (statsmodels, matrix)** | W2 c13–15, c38–42 | `sm.add_constant`, `sm.OLS(y, X).fit()`, `.summary()` | see below |
| **OLS (statsmodels formula)** | W2 c28 | `smf.ols(formula=..., data=...)` | `smf.ols(formula = "Price ~ SQFT", data = br)` |
| Prediction | W2 c56–60 | `results.predict(...)` | `results_formula.predict( new_house_df )` |
| **Train/test split** | W3 c2; W5 c7 | `df.sample(frac=0.7, random_state=1)` + `isin` | `test = data[data.index.isin(train.index)==False]` |
| **KNN** | W3 c5; W5 c10 | `KNeighborsRegressor(n_neighbors=k)` | `knn.fit(train[['Limit']], train['Balance'])` |
| **KNN with several predictors (scaling)** | W5 c12, c29, c33 | `metric='mahalanobis', metric_params={'V': train[predictors].cov()}` | see below |
| **MSE / RMSE** | W3 c5; W5 c10 | `mean_squared_error(y_true, y_pred)`; `np.sqrt` for RMSE | `rmse = np.sqrt(mean_squared_error(test['Balance'], predictions))` |
| **K-fold CV iterator** | W5 c15 | `KFold(10)`; `KFold(n, shuffle=True, random_state=...)` | `kf=KFold(10)` |
| LOOCV | W5 c18 | `LeaveOneOut()` | `loo=LeaveOneOut()` |
| **CV score** | W5 c20–26 | `cross_val_score(model, X, y, cv=kf or 10, scoring='neg_mean_squared_error')` | see below |
| Custom scorer | W5 c24 | `make_scorer(mean_squared_error, greater_is_better=False)` | `scorer = make_scorer(...)` |
| **Tuning k by CV** | W5 c28–29, c33 | manual `for` loop + `cross_val_score`; best via `np.argmin` or `if cv_score >= best_score` | see below |
| Predictor choice for KNN | W5 c35–41 | compare CV error across predictor lists | `predictors=['Limit','Income', 'Student']` |
| Bias–variance (simulation) | W4 c2–8 | `np.random.seed(0)`, `np.random.normal`, KNN at a fixed `x_0` | conceptual only |
| Interactions, polynomial terms, AIC/BIC, subset/forward/backward selection, ridge, lasso, elastic net, `StandardScaler` | — | — | **Not yet available (Weeks 6–7 needed)** |

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

K-fold CV pattern (W5):
```python
from sklearn.model_selection import KFold
kf=KFold(10)
# Syntax: KFold(number of folds, shuffle=True/False, random_state = a number, if shuffling)

from sklearn.model_selection import cross_val_score
knn = KNeighborsRegressor(n_neighbors= 5)
scores = cross_val_score(knn, train[['Limit']], train['Balance'], cv=kf, scoring = 'neg_mean_squared_error')
```
W5 notes that `KFold` does not shuffle by default, and that the training set was already randomly permuted by `df.sample`.
Our `WeeklyRent_training.csv` is **not** pre-shuffled, so we use `KFold(10, shuffle=True, random_state=...)`
(the syntax shown in the W5 comment).

CV tuning loop (W5 c29):
```python
predictors=['Limit','Income']

neighbours=np.arange(1, 101)
cv_rmse = []

for k in neighbours:
    knn = KNeighborsRegressor(n_neighbors = k, metric='mahalanobis', metric_params={'V': train[predictors].cov()})
    scores = cross_val_score(knn, train[predictors], train['Balance'], cv=10, scoring = 'neg_mean_squared_error')
    rmse = np.sqrt(-1*np.mean(scores)) # taking the average MSE across folds, then taking the square root
    cv_rmse.append(rmse)

fig, ax= plt.subplots()
ax.plot(neighbours, cv_rmse, color='#FF7F0E', label='CV error')
ax.set_xlabel('Number of neighbours')
ax.set_ylabel('RMSE')
plt.legend()
plt.show()

print('Lowest CV error: K = {}'.format(1 + np.argmin(cv_rmse)))
```

Helper function (W5 c33) — the only user-defined function in the tutorials so far:
```python
def knn_test(predictors, response):
    neighbours=np.arange(1, 51)
    best_score = -np.inf
    for k in neighbours:
        knn = KNeighborsRegressor(n_neighbors = k, metric='mahalanobis', metric_params={'V': train[predictors].cov()})
        scores = cross_val_score(knn, train[predictors], train[response], cv=10, scoring = 'neg_mean_squared_error')
        cv_score = np.mean(scores)
        if cv_score >= best_score:
            best_score = cv_score
            best_knn = knn
    knn = best_knn
    knn.fit(train[predictors], train[response])
    ...
```

## 3. Coding style

- **Variable naming:** main DataFrame `data` (W5), `br` (W2) or `df` (W1); `train` / `test` DataFrames;
  `predictors` = list of column names, `response` = target name; columns selected inline
  (`train[predictors]`, `train['Balance']`) rather than `X_train` / `y_train`.
  statsmodels: `model_*` (unfitted) / `results_*` (fitted).
- **Pipeline vs manual:** manual, step by step. No `Pipeline`, `GridSearchCV` or `StandardScaler` so far.
  One small helper function (`knn_test`) in W5.
- **Scaling for KNN:** W5 handles differing scales with the **Mahalanobis distance**
  (`metric_params={'V': train[predictors].cov()}`), not `StandardScaler`.
- **MSE:** `mean_squared_error(y_true, y_pred)`; CV MSE = `-np.mean(scores)` from `cross_val_score(..., scoring='neg_mean_squared_error')`.
  Tutorials usually report RMSE (`np.sqrt`); the assignment needs **MSE**, so we will not take the square root.
- **Plots:** matplotlib. Two styles:
  `fig = plt.figure()` + `plt.xlabel/ylabel/title` + (`plt.savefig`) + `plt.show()` (W1, W2, W4), and
  `fig, ax = plt.subplots()` + `ax.set_xlabel/ax.set_ylabel` + `plt.legend()` + `plt.show()` (W3, W5).
  Default figure size; scatter with `alpha`; colours `'#1F77B4'` (test error) and `'#FF7F0E'` (CV error) in W5.
- **Printing numbers:** `.format` style (`'Lowest CV error: K = {}'.format(...)`, `"{0:.2f}".format(...)`),
  W4 also uses f-strings; `.round(2)`. We use 4 d.p.
- **Markdown / comments:** W2 and W5 put a markdown cell (short heading + 1–3 sentences on what and why) before most code;
  comments explain syntax (`# Syntax: KFold(number of folds, ...)`) or intent (`# taking the average MSE across folds`).
- **Tuning loops:** manual `for` loop over `np.arange`, append to list, pick with `np.argmin`
  (or track `best_score` in an `if`). No `GridSearchCV`.
- **Seeds:** `random_state=1` for the split (W3, W5); `np.random.seed(0)` for simulation (W4).

## 4. Gaps (so far)

- Still no tutorial code for: interactions and polynomial terms (beyond the simulated quadratic in W4),
  AIC/BIC, best subset / forward / backward selection, ridge, lasso, elastic net, `StandardScaler`
  (Weeks 6–7 needed).
- Lecture method list pending `QBUS2820_A1_context.md` §4.
