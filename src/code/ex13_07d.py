from itertools import combinations

import pandas as pd
import statsmodels.api as sm

from ex13_06a import load_car_mileage_data
from ex13_07b import create_mallows_cp

endog, exog = load_car_mileage_data()
vars_ = exog.columns

mallows_cp = create_mallows_cp(endog, exog)

data = {"Covariates": [], "Cp": [], "BIC": []}

k = len(vars_)
for r in range(1, k + 1):
    for covars in combinations(vars_, r):
        subset_exog = exog[list(covars)]
        model_res = sm.OLS(endog, subset_exog).fit()
        data["Covariates"].append(" ".join(covars))
        data["Cp"].append(mallows_cp(model_res))
        data["BIC"].append(model_res.bic)

results_df = pd.DataFrame(data)
results_df.sort_values("Cp", inplace=True, ignore_index=True)
pd.set_option("display.precision", 2)
print(results_df)
