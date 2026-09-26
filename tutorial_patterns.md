# Tutorial Patterns (Phase 0)

> Status: **Weeks 1–7 read.** Lecture slides (Weeks 1–7) are now in `lectures/` and summarised in
> `lecture_summary.md`, which serves as the method whitelist. Every method in the Gaps table below is covered
> in the lectures (interactions: W2; subset selection: W4–5; ridge/elastic net: W6–7), so they are allowed.

Files read:

| File | Cells | Content |
|---|---|---|
| `week01/Tutorial_01_task_1_solution.ipynb` | 3 | Load Excel, line plot of average spend by year |
| `week01/Tutorial_01_task_2_solution.ipynb` | 5 | Histograms (all / subset / two groups overlaid), group mean & variance |
| `week02/Tutorial_02.ipynb` | 61 | Dummies, correlation, scatter, SLR/MLR with `statsmodels`, matrix OLS, formula API, plotting the fit, prediction |
| `week02/Tutorial_02_task.ipynb` | 2 | Task only (OLS of `AmountSpent`) |
| `week03/Tutorial_03_task_solution.ipynb` | 9 | 70/30 train/test split, KNN, loop over k, test RMSE vs k |
| `week04/Tutorial_04_task_solution.ipynb` | 10 | Simulated quadratic data; bias–variance of KNN (k = 1 vs 50) |
| `week05/Tutorial_05.ipynb` | 43 | Dummies, `KFold`, LOOCV, `cross_val_score`, CV loop for k, KNN with Mahalanobis distance, `knn_test` helper |
| `week06/Tutorial_06.ipynb` | 34 | `GridSearchCV` for KNN, `PolynomialFeatures` + `LinearRegression` with CV over degree, comparing models on the same `KFold`, AIC/BIC by formula |
| `week07/Tutorial_07.ipynb` | 42 | Manual standardisation (train mean/std), OLS backward elimination by p-value checking AIC/BIC, `Lasso`, `LassoCV`, refit OLS on Lasso-selected predictors |

Data files (`BatonRouge.xls`, `Credit.csv`, `CustomerAcquisition.xls`, …) are stored next to the notebooks that use them.

---

## 1. Libraries and imports

| Library | Import as used in tutorials | First seen | Used for |
|---|---|---|---|
| pandas | `import pandas as pd` | W1 | `read_excel`, `read_csv(index_col=...)`, `get_dummies`, `.corr()`, result tables |
| NumPy | `import numpy as np` | W1 | `arange`, `linspace`, `argmin`/`argmax`, `mean`, `sqrt`, `log`, `ravel`, `random.seed` |
| Matplotlib | `import matplotlib.pyplot as plt` | W1 | all plots |
| seaborn | `import seaborn as sns` | W2 | `pairplot`; `set_context('notebook')`, `set_style('ticks')` |
| statsmodels | `import statsmodels.api as sm`; `import statsmodels.formula.api as smf` | W2 | `sm.add_constant`, `sm.OLS`, `.summary()` (incl. AIC/BIC); `smf.ols` |
| warnings | `import warnings`; `warnings.filterwarnings('ignore')` | W7 | silence warnings |
| scikit-learn | `from sklearn.neighbors import KNeighborsRegressor` | W3 | KNN |
| | `from sklearn.metrics import mean_squared_error, make_scorer` | W3/W5 | MSE, custom scorer |
| | `from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score` | W5 | CV |
| | `from sklearn.model_selection import GridSearchCV, ParameterGrid` | W6 | grid search over hyperparameters |
| | `from sklearn.preprocessing import PolynomialFeatures` | W6 | polynomial basis |
| | `from sklearn.linear_model import LinearRegression` | W6 | OLS inside CV |
| | `from sklearn import linear_model` → `linear_model.Lasso`; `from sklearn.linear_model import LassoCV` | W7 | Lasso |

First-cell convention (W5, W7):
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
sns.set_context('notebook')
sns.set_style('ticks')
```
Other packages are imported *"in context to make clear what we are using them for"* (W5).

Not used in any tutorial: `Pipeline`, `ColumnTransformer`, `StandardScaler` (only mentioned in W7 as an alternative),
`Ridge`/`RidgeCV`, `ElasticNet`, `SplineTransformer`, trees/forests/boosting, `train_test_split`.

## 2. Method → tutorial code map

| Method | Where | Functions & classes | Excerpt |
|---|---|---|---|
| OLS — statsmodels matrix | W2 c13–15, c38–42; W7 c24 | `sm.add_constant(X, prepend=True)`, `sm.OLS(y, X).fit()`, `.summary()` | see A |
| OLS — statsmodels formula | W2 c28 | `smf.ols(formula=..., data=...).fit()` | `smf.ols(formula = "Price ~ SQFT", data = br)` |
| OLS — sklearn (for CV) | W6 c16, c25 | `LinearRegression()` + `cross_val_score` | see D |
| Dummies | W2 c5–7; W5 c5 | `pd.get_dummies(s, drop_first=True)`; `(col == value).astype(int)` | `data['Student']=(data['Student'] =='Yes').astype(int)` |
| Interactions | — | **no tutorial code** | — |
| Polynomial terms | W6 c16, c25, c31; W7 data | `PolynomialFeatures(k).fit_transform(x.reshape(-1,1))` (one predictor); W7 uses ready-made squared columns (`Acq_Expense_SQ`) | see D |
| KNN | W3 c5; W5 c10–12; W6 c5 | `KNeighborsRegressor(n_neighbors=k)`; multi-predictor with `metric='mahalanobis', metric_params={'V': train[predictors].cov()}` | see C |
| Train / test split | W3 c2; W5 c7; W7 c12 | `data.sample(frac=..., random_state=1)` + `isin` | `test = data[data.index.isin(train.index)==False].copy()` |
| K-fold CV | W5 c15–28; W6 c16, c26 | `KFold(10)` / `KFold(n, shuffle=True, random_state=...)`, `cross_val_score(..., scoring='neg_mean_squared_error')` | see B |
| Same folds for all models | W6 c24–27 | one `kf = KFold(3)` object passed to every `cross_val_score` | *"To ensure that each model uses the same K-Folds … store the K-Fold split into a Python object"* |
| Hyperparameter search | W5 c28–33 (loop); W6 c5–8 (`GridSearchCV`) | manual loop + `np.argmin`; or `GridSearchCV(model, param_grid)`, `.best_params_`, `.best_estimator_`, `.cv_results_` | see C |
| AIC / BIC | W6 c31 (formula); W7 c24–30 (`summary()`) | `n*np.log(sigma2_hat)+2*d`, `n*np.log(sigma2_hat)+np.log(n)*d`; statsmodels reports AIC/BIC | see E |
| Backward elimination | W7 c24–31 | manual: drop the largest-p-value predictor, refit `sm.OLS`, keep if AIC and BIC fall | see F |
| Best subset / forward stepwise | — | **no tutorial code** | — |
| Standardisation | W7 c14–18 | manual: `mu=train[predictors].mean()`, `sigma=train[predictors].std()`, apply to train **and** test | see G |
| Lasso | W7 c33–39 | `linear_model.Lasso(alpha=...)`, `LassoCV(cv=10)`, `lasso.alpha_`, `lasso.coef_`, `np.ravel(train[response])` | see H |
| Refit OLS on Lasso-selected predictors | W7 c39 | list comprehension over `lasso.coef_ != 0`, then `sm.OLS` | see H |
| Ridge | — | **no tutorial code** (named in W7 markdown only; Sherry believes Ridge code was provided — awaiting the file) | — |
| Elastic net | — | **no tutorial code** (named in W7 markdown only) | — |
| MSE | W3 c5; W5; W6 c31 | `mean_squared_error(y_true, y_pred)`; CV MSE = `-np.mean(scores)` | — |
| Bias–variance (simulation) | W4 | `np.random.seed(0)`, `np.random.normal` | conceptual |

**A. OLS (W7 c24)**
```python
import statsmodels.api as sm

x_with_intercept = sm.add_constant(train[predictors], prepend=True)
ols = sm.OLS(train[response],x_with_intercept)
est = ols.fit()
print(est.summary())
```

**B. K-fold CV (W5 c15, c20)**
```python
from sklearn.model_selection import KFold
kf=KFold(10)
# Syntax: KFold(number of folds, shuffle=True/False, random_state = a number, if shuffling)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, train[['Limit']], train['Balance'], cv=kf, scoring = 'neg_mean_squared_error')
```

**C. KNN tuning (W5 c29 loop; W6 c5 GridSearchCV)**
```python
neighbours=np.arange(1, 101)
cv_rmse = []
for k in neighbours:
    knn = KNeighborsRegressor(n_neighbors = k, metric='mahalanobis', metric_params={'V': train[predictors].cov()})
    scores = cross_val_score(knn, train[predictors], train['Balance'], cv=10, scoring = 'neg_mean_squared_error')
    rmse = np.sqrt(-1*np.mean(scores)) # taking the average MSE across folds, then taking the square root
    cv_rmse.append(rmse)
print('Lowest CV error: K = {}'.format(1 + np.argmin(cv_rmse)))
```
```python
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_neighbors': np.arange(1,11)
}
model = KNeighborsRegressor()
grid_cv_obj = GridSearchCV(model, param_grid)
grid_cv_obj.fit(x.reshape(-1, 1), y_train)
best_model = grid_cv_obj.best_estimator_
print(grid_cv_obj.best_params_)
results = pd.DataFrame(grid_cv_obj.cv_results_)
```

**D. Polynomial degree by CV (W6 c16) and model comparison table (W6 c26–27)**
```python
max_deg = 10
cv_scores_lr = []
for i in range(1, max_deg):
    poly_transformer = PolynomialFeatures(i)
    poly_x = poly_transformer.fit_transform(x.reshape(-1,1))
    lin_reg = LinearRegression()
    scores_lr = cross_val_score(lin_reg, poly_x, y_train, cv=3, scoring = 'neg_mean_squared_error')
    cv_score = np.mean(scores_lr)
    cv_scores_lr.append(cv_score)
1 + np.argmax(cv_scores_lr)
```
```python
data = [
    {"Model": "Polynomial Regression", "CV score": poly_mean},
    {"Model": "KNN Regressor", "CV score": knn_mean}
]
results = pd.DataFrame(data)
results = results.set_index("Model")
```

**E. AIC / BIC (W6 c31)**
```python
y_pred = lin_reg.predict(poly_x)
sigma2_hat = mean_squared_error(y, y_pred)
number_of_parameters = k+2
aic_value = n*np.log(sigma2_hat)+2*number_of_parameters
bic_value = n*np.log(sigma2_hat)+np.log(n)*number_of_parameters
```

**F. Backward elimination (W7 c25–31)** — edit the predictor list by commenting out one name, refit, compare AIC/BIC:
```python
predictors_new = ['First_Purchase',
# 'Acq_Expense',
 'Acq_Expense_SQ',
 ...]
x_with_intercept = sm.add_constant(train[predictors_new], prepend=True)
ols = sm.OLS(train[response],x_with_intercept)
est = ols.fit()
print(est.summary())
```
Markdown: *"As the p-value for `Revenue` is the largest (larger than 0.05), let's remove this predictor"*;
*"Both AIC and BIC got smaller, hence we should remove `Acq_Expense`."*

**G. Standardisation (W7 c14–18)**
```python
mu=train[predictors].mean() # mean for each feature
sigma=train[predictors].std() # std for each feature
train[predictors]=(train[predictors]-mu)/sigma
test[predictors]=(test[predictors]-mu)/sigma
```
*"It's crucial to use the mean and standard deviation calculated from the training data."*
The response is not standardised.

**H. Lasso (W7 c33–39)**
```python
from sklearn.linear_model import LassoCV

lasso = LassoCV(cv=10)
lasso.fit(train[predictors], np.ravel(train[response])) # the np.ravel is a necessary detail for compatibility
print("LASSO Lambda: {0}".format(lasso.alpha_))
pd.DataFrame(lasso.coef_.round(3), index = predictors).T

selected_predictors = [predictors[i] for i, coef in enumerate(lasso.coef_) if coef != 0]
```

## 3. Coding style

- **Variable naming:** DataFrame `data`; splits `train` / `test`; `response = ['CLV']` (W7, a list) or `'Balance'` (W5, a string);
  `predictors` = list of column names built with a list comprehension (W7 c8); columns selected inline
  (`train[predictors]`) — no `X_train` / `y_train`. statsmodels objects `ols` / `est` (W7) or `model_*` / `results_*` (W2).
  sklearn objects: `knn`, `lin_reg`, `poly_reg`, `lasso`, `grid_cv_obj`.
- **Pipeline vs manual:** always manual. No `Pipeline`. Transformations are applied explicitly
  (`poly_transformer.fit_transform(...)`, manual standardisation).
- **Where preprocessing is fitted:** W7 standardises the whole training set once with training mean/std,
  then runs `LassoCV(cv=10)` on it (scaling is not re-fitted inside each fold).
- **MSE:** `mean_squared_error(y_true, y_pred)`; CV MSE = `-np.mean(scores)` from `scoring='neg_mean_squared_error'`.
  Tutorials often report RMSE (`np.sqrt`); the assignment needs MSE.
- **Plots:** matplotlib, default figure size, one figure per cell. Either
  `fig = plt.figure()` + `plt.xlabel/ylabel/title` + `plt.legend(...)` + `plt.show()`, or
  `fig, ax = plt.subplots()` + `ax.set_xlabel/ax.set_ylabel` + `plt.legend()` + `plt.show()`.
  Scatter with `alpha`; truth line vs points (W4, W6); CV curve colour `'#FF7F0E'`. `plt.savefig("part1.jpg")` in W1.
- **Results tables:** list of dicts → `pd.DataFrame(...).set_index("Model")` (W6 c27); coefficients shown as
  `pd.DataFrame(coef.round(3), index = predictors).T` (W7).
- **Markdown / comments:** a markdown cell before most code cells — a heading and 1–3 sentences on what and why,
  sometimes LaTeX formulas and bolded key points; interpretation cells after results
  (*"Both AIC and BIC got smaller, hence …"*). Comments are short and explain syntax or intent.
- **Tuning:** manual `for` loops over `np.arange` / `range`, append to list, choose with `np.argmin` / `np.argmax`;
  `GridSearchCV` from W6; `LassoCV(cv=10)` for Lasso.
- **Printing:** `.format` (`"LASSO Lambda: {0}".format(...)`), f-strings in W4, `.round(...)`. We use 4 d.p.
- **Seeds:** `random_state=1` for splits; `np.random.seed(0)` for simulation.

## 4. Gaps (methods with no tutorial code)

| Method | Status in tutorials | Proposed minimal approach (to confirm) |
|---|---|---|
| Interactions | not shown | add product columns by hand in `make_features(df)` (e.g. `df['Dist_x_HighDemand'] = df['DistanceCBD'] * df['HighDemandArea']`), in the same way W7 uses ready-made `_SQ` columns; or `PolynomialFeatures(2)` on several columns (W6 uses it on one) |
| Best subset / forward stepwise | not shown (only manual backward elimination) | manual backward elimination as in W7 (p-value + AIC/BIC), optionally a short loop that automates the same steps |
| Ridge | named only | `linear_model.Ridge` / `RidgeCV(cv=10)` used exactly like `Lasso` / `LassoCV` in W7 |
| Elastic net | named only | `ElasticNetCV(cv=10)` used like `LassoCV` |
| `StandardScaler` | named only | not needed: use the W7 manual `mu` / `sigma` standardisation |
| Scaling inside CV folds | not done in W7 (W7 standardises the whole training set once) | **Decided (2026-09-26): use a scikit-learn `Pipeline` so that standardisation is re-fitted inside each CV fold.** Marked `# NOTE: not from tutorial` in the notebook. |

Methods in the current scaffold that do **not** appear in any tutorial and will be removed under CLAUDE.md §3.2:
`DecisionTreeRegressor`, `RandomForestRegressor`, `HistGradientBoostingRegressor`, `SplineTransformer`,
`ColumnTransformer`, `FunctionTransformer`, `DummyRegressor`.
(`Pipeline` + `StandardScaler` are kept only for standardisation inside CV folds, per the decision above.)

## 5. EDA-specific deviations (Phase 1)

| Item | Why | Tutorial basis |
|---|---|---|
| `plt.boxplot` for rent by group | box plots requested in CLAUDE.md Phase 1 | W1 compares groups with overlaid histograms |
| `sns.heatmap` for correlations | heatmap requested in CLAUDE.md Phase 1 | W2/W5 print `.corr()`; seaborn used in W2 |
| `plt.barh` for correlations with rent | bar chart requested to rank correlations with the response | W5 prints `train.corr().round(3)['Balance']` |
