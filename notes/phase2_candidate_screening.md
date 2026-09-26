# Candidate-term screening (moved from EDA Section 2.9, to be placed at the start of the modelling section)

Decision (2026-09-26): AIC screening is a modelling step, so it was removed from the EDA and will be reused in Phase 2
to choose the interaction and squared terms for M2. Results at the time of removal: largest AIC drops were
DistanceCBD^2 787.16, Bedrooms x HighDemandArea 499.67, FloorArea x HighDemandArea 318.75, Bedrooms x FloorArea 302.76,
FloorArea^2 262.02, Bedrooms^2 217.44; terms involving PropertyAge ~0.

```python
# Base model: OLS with the seven predictors (linear terms only)
x_with_intercept = sm.add_constant(train[predictors], prepend=True)
base = sm.OLS(train[response], x_with_intercept).fit()
print('AIC of the base model: {0:.4f}'.format(base.aic))
```

```python
# Add each pairwise interaction to the base model, one at a time
interaction_results = []

for j in range(len(predictors)):
    for k in range(j + 1, len(predictors)):     # k starts at j + 1, so each pair is used once and never with itself
        var1 = predictors[j]
        var2 = predictors[k]

        x = train[predictors].copy()               # copy, so the original data are not changed
        x['new_term'] = x[var1] * x[var2]          # interaction term = product of the two predictors
        x_with_intercept = sm.add_constant(x, prepend=True)
        est = sm.OLS(train[response], x_with_intercept).fit()

        interaction_results.append({'Term': var1 + ' x ' + var2,
                                    'AIC drop': base.aic - est.aic,          # how much the AIC falls
                                    'p-value': est.pvalues['new_term']})     # significance of the new term

# Put the results in a table, sorted from the most to the least useful term
interaction_results = pd.DataFrame(interaction_results)
interaction_results = interaction_results.sort_values('AIC drop', ascending = False).set_index('Term')
interaction_results.round(4)
```

```python
# For comparison, add the square of each non-binary predictor, one at a time
# (0/1 variables are skipped, because 0^2 = 0 and 1^2 = 1)
square_results = []

for var in ['DistanceCBD', 'FloorArea', 'Bedrooms', 'PropertyAge']:
    x = train[predictors].copy()
    x['new_term'] = x[var] ** 2                    # squared term
    x_with_intercept = sm.add_constant(x, prepend=True)
    est = sm.OLS(train[response], x_with_intercept).fit()

    square_results.append({'Term': var + '^2',
                           'AIC drop': base.aic - est.aic,
                           'p-value': est.pvalues['new_term']})

square_results = pd.DataFrame(square_results)
square_results = square_results.sort_values('AIC drop', ascending = False).set_index('Term')
square_results.round(4)
```
