# Results Log — Phase 2 (model comparison)

Workflow (Tutorials 5–6, Lectures 4–5): each model is built and **immediately evaluated with the
same 10-fold CV** (`KFold(10, shuffle=True, random_state=1)`); all CV MSEs are compared in one table. CV MSE = mean of the 10 fold MSEs;
SE = standard deviation of the fold MSEs / sqrt(10). Standardisation (ridge, lasso, KNN) is inside a
`Pipeline`, so it is re-fitted on the training part of every fold; hyperparameters (λ, k) are chosen by a manual
CV loop over a grid on the same folds (no inner CV, no leakage). Stepwise selection is done on all training data before CV
(as in the lectures); a nested check earlier showed the effect is small (BIC models unchanged, AIC +4).

Columns from `make_features(df)`: 7 predictors + 4 squared (`_SQ`) + 21 interactions (`_x_`) = 32.

| ID | Model | Features | Hyperparameters | CV MSE | SE | Comment |
|---|---|---|---|---|---|---|
| M0 | Null model | none (training mean) | – | 46409.4061 | 842.9485 | benchmark |
| M1 | OLS | 7 linear terms | – | 3302.3821 | 29.6067 | |
| M2 | Best subset (7 predictors, 128 subsets) | best size by CV = all 7 | size = 7 | 3302.3821 | 29.6067 | every predictor is useful; identical to M1 |
| M3 | Forward stepwise (AIC) | 7 + 7: DistanceCBD_SQ, Bedrooms_x_HighDemandArea, FloorArea_SQ, DistanceCBD_x_NearTrain, Bedrooms_x_Furnished, PropertyAge_SQ, Furnished_x_HighDemandArea | – | 2005.8861 | 26.4687 | same terms as backward (AIC) |
| **M3** | **Forward stepwise (BIC)** | **7 + 6: the above without Furnished_x_HighDemandArea** | – | **2005.6203** | 27.4624 | same terms as backward (BIC); lowest CV MSE |
| M4 | Backward stepwise (AIC) | same 7 extra terms as forward (AIC) | – | 2005.8861 | 26.4687 | |
| **M4** | **Backward stepwise (BIC)** | same 6 extra terms as forward (BIC) | – | **2005.6203** | 27.4624 | lowest CV MSE |
| M5 | Lasso (standardised) | 32 columns | α = 0.0788 | 2012.2895 | 25.5644 | |
| M5 | Ridge (standardised) | 32 columns | α = 0.3857 | 2017.8831 | 27.2966 | |
| M6 | KNN (standardised, Euclidean) | 7 predictors | k = 7 | 3438.6270 | 69.4793 | worse than linear OLS |
