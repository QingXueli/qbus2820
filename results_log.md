# Results Log — Phase 2 (model comparison)

All models use the same `KFold(10, shuffle=True, random_state=1)` on the 5000 training rows. CV MSE = mean of the
10 fold MSEs; SE = standard deviation of the fold MSEs / sqrt(10). For M2 and M3 the term selection is repeated inside
each fold (nested), so the validation fold is never used to choose terms. For M4 the standardisation and the choice of
λ are inside a Pipeline (re-fitted in each fold). Candidate columns come from `make_features(df)`: 7 predictors +
4 squared terms (`_SQ`) + 21 pairwise interactions (`_x_`) = 32 columns.

| ID | Model | Features | Hyperparameters | CV MSE | SE | Comment |
|---|---|---|---|---|---|---|
| M0 | Null model | none (training mean) | – | 46409.4061 | 842.9485 | benchmark |
| M1 | OLS | 7 linear terms | – | 3302.3821 | 29.6067 | misses curvature and interactions |
| M2 | OLS | 7 + terms with AIC drop > 100 when added one at a time (11 on full data) | threshold 100 | 2029.9416 | 20.3766 | screening is slightly optimistic without nesting (2016.86) |
| M3 | OLS, backward stepwise (AIC) | 7 + 7 selected on full data: DistanceCBD_SQ, FloorArea_SQ, PropertyAge_SQ, DistanceCBD_x_NearTrain, Bedrooms_x_Furnished, Bedrooms_x_HighDemandArea, Furnished_x_HighDemandArea | criterion AIC | 2009.9860 | 26.9744 | 6–8 terms kept across folds |
| **M3** | **OLS, backward stepwise (BIC)** | **7 + 6 selected: DistanceCBD_SQ, FloorArea_SQ, PropertyAge_SQ, DistanceCBD_x_NearTrain, Bedrooms_x_Furnished, Bedrooms_x_HighDemandArea** | criterion BIC | **2005.6203** | 27.4624 | lowest CV MSE; same 6 terms chosen in every fold |
| M4 | Lasso (standardised) | 32 columns | α = 0.0720 (CV) | 2012.1438 | 25.6644 | |
| M4 | Elastic net (standardised) | 32 columns | α = 0.0720, l1_ratio = 1 (CV) | 2012.1438 | 25.6644 | CV chose l1_ratio = 1, i.e. the lasso |
| M4 | Ridge (standardised) | 32 columns | α = 0.5179 (CV) | 2018.0675 | 27.2604 | |
| M5 | KNN (standardised, Euclidean) | 7 predictors | k = 7 (CV over 1–50) | 3438.6270 | 69.4793 | worse than linear OLS |
