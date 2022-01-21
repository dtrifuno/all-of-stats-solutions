import pathlib

import pandas as pd
import statsmodels.api as sm

src_dir = pathlib.Path(__file__).resolve().parent.parent
dat_path = src_dir.joinpath("data", "coris.dat")
df = pd.read_table(dat_path, sep=",", skiprows=[1, 2, 3], index_col=0)
exog = df[df.columns[:-1]]
endog = df[df.columns[-1:]]

full_model = sm.Logit(endog, sm.add_constant(exog)).fit(disp=False)
best_model = full_model
remaining_vars = set(exog.columns)
while True:
    best_model_run = None
    for var in remaining_vars:
        candidate_vars = remaining_vars - set([var])
        new_exog = exog[list(candidate_vars)]

        model = sm.Logit(endog, new_exog, offset=True).fit(disp=False)
        if best_model_run is None or model.aic < best_model_run.aic:
            best_model_run = model

    if best_model_run.aic < best_model.aic:
        best_model = best_model_run
        remaining_vars = set(best_model.params.index) - set(["const"])
    else:
        break

print(best_model.summary())
