# Report Notes (material for the report — not the report itself)

These notes hold the details that were moved out of the notebook markdown. Numbers are from the notebook (4 d.p. where
the assignment requires). Sherry writes the report prose herself.

---

## 1. Data and EDA

### Key facts
- 5000 training rows, 1000 test rows; all numeric; no missing values or duplicates.
- Binary predictors (`NearTrain`, `Furnished`, `HighDemandArea`) already 0/1 dummies → no encoding.
- `DistanceCBD` max is exactly 40 km and `FloorArea` min exactly 35 m² (possible capping; rows kept).
- `PropertyAge` has a long right tail (max 100 years) but values are plausible.

### Train vs test (Section 2.4)
- Means, standard deviations and ranges of test covariates are very close to training (e.g. mean `DistanceCBD`
  13.5099 vs 13.6192); every test value lies inside the training range.
- Reasoning: the model will not extrapolate, so CV error on the training data is a reasonable guide to test error.

### Response (Section 2.5)
- Mean 860.6811, median 849.0900, std 215.4138, skewness 0.2660 → roughly symmetric, bell-shaped.
- Reasoning: no log transform; model on the original AUD scale, which is also the scale of the test MSE.

### Non-linearity (Section 2.6)
- Linear vs quadratic fits: distance curve flattens beyond ~25 km; floor-area curve rises at a decreasing rate;
  `PropertyAge` fits almost coincide.
- Reasoning: a straight line assumes a constant slope; squared terms let the slope change (Lecture 2 "quadratic effects").
  Tutorials 4 and 6 use a quadratic data-generating process; Tutorial 7 data include `_SQ` columns.

### Group differences (Section 2.7)
- Mean rent by bedrooms: 717.9369, 823.7616, 938.8978, 1020.8464, 1087.7315 (increments +106, +115, +82, +67).
- Binary groups (1 vs 0): HighDemandArea 981.6058 vs 759.2359 (+222); NearTrain 934.5973 vs 797.2066 (+137);
  Furnished 918.5214 vs 835.9160 (+83).
- Raw differences do not control for other predictors: the OLS coefficient for `HighDemandArea` is 161.6766, smaller
  than 222, because high-demand areas are also closer to the CBD (correlation −0.3727).

### Correlations (Section 2.8)
- With rent: HighDemandArea 0.5142, Bedrooms 0.5062, FloorArea 0.3567, NearTrain 0.3180, Furnished 0.1757,
  PropertyAge −0.1375, DistanceCBD −0.4421.
- `Bedrooms`–`FloorArea` 0.8821 (multicollinearity; Lecture 6–7: unstable OLS coefficients; ridge handles it).
  Other predictor pairs |r| ≤ 0.53.

### Interactions (Section 2.9)
- Definition (Lecture 2): the effect of one predictor depends on another; e.g. effect of an extra bedroom is b1 in normal
  areas and b1 + b3 in high-demand areas. Lecture 2: finding non-linear effects "requires domain-knowledge, or trial and error".
- Group slopes: Bedrooms 97.96 vs 138.86; FloorArea 2.78 vs 4.26 (by HighDemandArea); DistanceCBD −6.53 vs −6.61
  (by HighDemandArea, parallel); DistanceCBD −9.72 vs −7.81 (by NearTrain).
- Limitation: group slopes do not hold other predictors fixed → tested formally in modelling.

### Baseline OLS and outliers (Section 2.10)
- R² = 0.929; all p < 0.001. Coefficients: const 481.9479, DistanceCBD −13.6208, Bedrooms 83.0277, FloorArea 2.9902,
  PropertyAge −1.7023, NearTrain 79.6334, Furnished 89.3712, HighDemandArea 161.6766.
- Residual skew 0.083, kurtosis 3.128 → roughly normal errors.
- Standardised residual = residual / σ̂ (σ̂ = 57.4 AUD). Under normal errors 0.27% exceed ±3 → 13.4990 expected;
  17 observed; largest 4.2356. Positive outliers mostly 30–40 km from the CBD, negative mostly five-bedroom properties
  → model misspecification (missing curvature), not data errors → all kept.

---

## 2. Modelling

### Validation design
- 10-fold CV, `KFold(10, shuffle=True, random_state=1)`; same folds for every model (Tutorial 6).
  Lecture 4–5: K is typically 5, 10 or n; choose the model with the smallest CV error; refit on all data.
- Shuffle: the training file is not in random order (Tutorial 5 notes KFold does not shuffle by default).
- SE of CV MSE = std of the 10 fold MSEs / √10. One-standard-error reasoning: models within 1 SE are treated as equal;
  prefer the simpler one.
- Standardisation only for ridge/lasso/elastic net/KNN (OLS is scale-invariant, Lecture 6–7). Scaler inside a
  `Pipeline` → mean/std computed on the training part of each fold only (no leakage). λ and k chosen with manual CV
  loops (Tutorial 5 style), not `LassoCV`/`RidgeCV`, so there is no inner CV that could leak.
- Why CV rather than training MSE: training MSE is optimistic (Lecture 4–5: RSS and R² are not appropriate for comparing
  models with different numbers of predictors). M1: training MSE 3289.4727 vs CV MSE 3302.3821 (little overfitting).

### Features
- `make_features(df)`: 7 predictors + 4 squared (`_SQ`) + 21 pairwise interactions (`_x_`) = 32 columns; used for
  train and test so the final cell can reuse it.
- Hierarchy principle (Lecture 2): main effects always kept whenever an interaction is included.

### Models and reasoning chain
| Model | Why | Result (CV MSE, SE) |
|---|---|---|
| M0 null | benchmark "no information" (Lecture 4–5 $M_0$) | 46409.4061 (842.9485) |
| M1 OLS, 7 terms | simplest regression | 3302.3821 (29.6067) |
| M2 best subset (128 subsets) | are all 7 predictors needed? (Lecture 4–5) | 3302.3821 — all 7 kept |
| M3 EDA-driven OLS | test the EDA hypotheses | 2075.4951 (27.7922); Bedrooms_SQ p = 0.6078 and FloorArea_x_HighDemandArea p = 0.4913 not significant (collinearity) |
| M4/M5 forward/backward (AIC) | systematic search over 25 extra terms (Lecture 4–5; Tutorial 7) | 2005.8861 (26.4687), 7 extra terms |
| **M4/M5 forward/backward (BIC)** | BIC penalises complexity more | **2005.6203 (27.4624)**, 6 extra terms |
| M6 ridge / lasso | shrink all 32 (Lecture 6–7); λ chosen by AIC(λ)/BIC(λ) with effective df (ridge: eigenvalue formula; lasso: non-zero count), 30-value grid 0.001–100 | lasso AIC 2012.4817 (25.8717), α = 0.0530, df 23; lasso BIC 2013.3032 (25.1854), α = 0.1172, df 20 (12 of 32 coefficients zero); ridge AIC 2017.8835 (27.1835), α = 0.5736; ridge BIC 2018.6737 (26.4942), α = 1.8874 (no zero coefficients) |
| M6 elastic net | no df formula in the lectures → λ by CV on a small grid (0.01–10, 10 values), l1_ratio = 0.9 | 2022.4385 (25.0940), α = 0.0100 (grid minimum) |
| M7 KNN | non-parametric alternative (Lecture 3) | 3438.6270 (69.4793), k = 7 |

- Best subset only feasible for the 7 predictors (2^32 subsets for 32 columns) → stepwise for the expanded set.
- Forward and backward reach the same terms → stable selection.
- `FloorArea_x_HighDemandArea` not selected although it looked strong in the plots: it overlaps with
  `Bedrooms_x_HighDemandArea` (Bedrooms–FloorArea r = 0.88).
- KNN worse than linear OLS: the relationship is mostly linear with smooth curvature; parametric models suit it
  (Lecture 3 comparison of KNN vs linear regression).
- Regularisation gives little gain: n = 5000 is large relative to 32 columns. Bias–variance check (ridge, scratch, not in
  the notebook): training MSE vs CV MSE = 1991.22 vs 2018.04 at λ = 0.001 (gap 26.81), 1991.38 vs 2017.88 at λ = 0.5736,
  2401.54 vs 2472.43 at λ = 100, 3800.09 vs 3907.53 at λ = 1000. The small gap at small λ means OLS has little variance to
  remove; larger λ mainly adds bias (Lectures 6–7: U-shaped test MSE, minimum here at the left end).

### Recommended final model
- OLS with 7 main effects + `DistanceCBD_SQ`, `FloorArea_SQ`, `PropertyAge_SQ`, `Bedrooms_x_HighDemandArea`,
  `Bedrooms_x_Furnished`, `DistanceCBD_x_NearTrain`. Lowest CV MSE, simplest among models within 1 SE, interpretable.
  (Pending Sherry's confirmation.)

---

## 3. Suggested figures and tables
| Item | Notebook section | Suggested caption |
|---|---|---|
| Rent histogram | 2.5 | Distribution of weekly rent (training data) |
| Rent vs continuous predictors (1×3) | 2.6 | Weekly rent against each continuous predictor with linear and quadratic fits |
| Box plots (1×4) | 2.7 | Weekly rent by number of bedrooms and binary predictors |
| Correlation heatmap / bar chart | 2.8 | Correlations between variables / with weekly rent |
| Interaction plots (1×4) | 2.9 | Separate linear fits by group: non-parallel lines suggest interactions |
| Model comparison table + bar chart | 3.10 | 10-fold CV MSE of all candidate models (error bars = 1 SE) |

---

## 4. Limitations (draft list)
- Stepwise selection uses all training data before CV, so its CV MSE may be slightly optimistic (a nested check earlier
  gave BIC unchanged, AIC 2009.99 vs 2005.89).
- Possible capping of `DistanceCBD` (40 km) and `FloorArea` (35 m²) is not modelled explicitly.
- Only pairwise interactions and squared terms were considered. Robustness check (scratch, not in the notebook): adding
  any of the 35 three-way interactions (with their two-way sub-terms, hierarchy principle) to the BIC model did not lower
  the CV MSE (best 2005.7548 vs 2005.6203; all differences far below 1 SE), so higher-order interactions are not used.
- Elastic net chose the smallest λ on the grid (boundary).
- Residual standard deviation ≈ 45 AUD may be close to the irreducible noise.
