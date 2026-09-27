# Lecture Summary (Weeks 1–7) — method whitelist for Assignment 1

Source: `lectures/QBUS2820-01.pdf`, `-02`, `-03`, `-0405`, `-0607`. This file stands in for §4 of the
`QBUS2820_A1_context.md` referred to in `CLAUDE.md` (not yet provided).

## Topics by lecture

| Lecture | Topics | Relevant to the assignment |
|---|---|---|
| W1 Introduction | supervised learning, regression function, squared-error loss, expected prediction error, under/overfitting, parametric vs non-parametric, prediction vs interpretability, no free lunch | framing; test MSE = squared-error loss |
| W2 Linear regression | SLR/MLR, least squares, MLE, t-tests and p-values, F-test, R², categorical covariates (dummies, baseline level), **interaction effects**, **hierarchy principle**, **quadratic effects**, non-linear effects | OLS, dummies, interactions, squared terms |
| W3 KNN | KNN regression, Euclidean / **normalised Euclidean** / **Mahalanobis** distance, choosing k, curse of dimensionality, comparison with linear regression | M5 (KNN with scaled predictors) |
| W4–5 Model selection | training/validation/test error, bias–variance decomposition, **AIC**, **BIC**, **K-fold CV / LOOCV**, **best subset**, **forward stepwise**, **backward stepwise** | model comparison, M3 (subset selection) |
| W6–7 Regularisation | multicollinearity, **ridge**, **lasso**, **elastic net**, group lasso, **standardisation** before ridge/lasso, selecting λ by AIC/BIC/CV, refit | M4 (ridge/lasso/elastic net) |

**Not covered (not allowed):** trees, random forests, boosting, splines/GAMs, SVR, neural networks
(the W2 slides say neural networks are "not covered in this unit").

## Key points for the modelling

- **Interactions (W2).** A term such as `Salary * Children` lets the effect of one predictor depend on another:
  effect of Salary = β1 + β3 × Children. **Hierarchy principle:** if an interaction is included, both main effects
  must be included.
- **Quadratic / non-linear effects (W2).** E.g. income = β0 + … + β2·age + β3·age². "Including appropriate non-linear
  effect terms into the regression model often increases the prediction accuracy. How? This requires
  **domain-knowledge, or trial and error**."
- **Removing predictors by p-value (W2).** Start from the full model, remove X_j if its p-value is large
  (typically > 0.1), repeat until all coefficients are significant.
- **AIC vs BIC (W4–5).** AIC = n log(RSS/n) + 2d, BIC = n log(RSS/n) + log(n)·d. "AIC aims to find the most
  predictive model"; BIC penalises complexity more and often chooses simpler models. RSS and R² alone are not
  appropriate for comparing models with different numbers of predictors.
- **Cross-validation (W4–5).** K = 5, 10 or n; choose the model with the smallest CV error; **refit on the entire
  dataset** after CV.
- **Best subset / FSS / BSS (W4–5).** Within each size choose by RSS; choose among sizes by CV error, AIC, BIC or
  adjusted R².
- **Multicollinearity (W6–7).** Strongly correlated columns make X'X nearly singular and the OLS estimates unstable
  (can even change sign). Ridge "handles multicollinearity well".
- **Standardisation (W6–7).** OLS is scale-invariant but ridge/lasso are not → standardise predictors first.
- **Selecting λ (W6–7).** Grid of λ; choose by AIC(λ), BIC(λ) or CV(λ); refit on all data
  (lasso: optionally refit OLS on the selected predictors).
- **Ridge vs lasso (W6–7).** No free lunch; lasso gives sparse, interpretable models; ridge suits correlated
  predictors that all have some effect. Elastic net is a compromise.
- **KNN scaling (W3).** Euclidean distance only makes sense if predictors are on the same scale → normalised
  Euclidean or Mahalanobis distance.

## Implications for our notebook

| Item | Lecture basis |
|---|---|
| Squared terms for `DistanceCBD`, `FloorArea`, `Bedrooms` | W2 quadratic / non-linear effects |
| Interaction terms, keeping main effects | W2 interaction effects, hierarchy principle |
| Screening candidate terms by AIC | W4–5 AIC ("most predictive model"); W2 "trial and error" |
| Final model choice by 10-fold CV MSE | W4–5 cross-validation |
| Ridge / lasso on standardised expanded features (elastic net not used) | W6–7 |
| KNN with Mahalanobis or normalised distance | W3 |
| Bedrooms–FloorArea correlation 0.88 | W6–7 multicollinearity → ridge |
