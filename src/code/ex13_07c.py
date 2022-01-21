import numpy as np
import statsmodels.api as sm

from ex13_06a import load_car_mileage_data

endog, exog = load_car_mileage_data()
model_res = sm.OLS(endog, exog).fit()
sigma = model_res.mse_resid

test_scores = np.abs(model_res.tvalues).sort_values(ascending=False)
k = len(test_scores)
n = model_res.nobs

min_score = float("inf")
min_score_model = None
for j in range(1, k + 1):
    vars_ = test_scores.index[:j]
    subset_exog = exog[vars_]
    j_model = sm.OLS(endog, subset_exog).fit()
    score = j_model.ssr + j * sigma * np.log(n)
    if score < min_score:
        min_score, min_score_model = score, vars_

print("Best Zheng-Loh model parameters:", *min_score_model)
