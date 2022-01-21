import pandas as pd
import statsmodels.api as sm

from ex13_06a import load_car_mileage_data


def create_mallows_cp(endog, exog):
    full_model = sm.OLS(endog, exog)
    full_model_res = full_model.fit()
    sigma = full_model_res.mse_resid
    return lambda res: res.ssr + 2 * len(res.params.index) * sigma


def forward_stepwise(endog, exog, loss_fn):
    best_model_res = None
    best_model_loss = float("inf")
    remaining_vars = set(exog.columns)
    current_vars = set()

    while True:
        best_model_run = None
        best_model_run_loss = float("inf")
        remaining_vars = remaining_vars - current_vars

        for var in remaining_vars:
            candidate_vars = current_vars | set([var])
            new_exog = exog[list(candidate_vars)]

            model_res = sm.OLS(endog, new_exog).fit()
            model_loss = loss_fn(model_res)
            if model_loss < best_model_run_loss:
                best_model_run = model_res
                best_model_run_loss = model_loss

        if best_model_run_loss < best_model_loss:
            best_model_res = best_model_run
            best_model_loss = best_model_run_loss
            current_vars = set(best_model_res.params.index)
        else:
            return best_model_res


def backward_stepwise(endog, exog, loss_fn):
    full_model = sm.OLS(endog, exog).fit()

    best_model = full_model
    best_model_loss = loss_fn(full_model)
    remaining_vars = set(exog.columns)
    while True:
        best_model_run = None
        best_model_run_loss = float("inf")
        for var in remaining_vars:
            candidate_vars = remaining_vars - set([var])
            new_exog = exog[list(candidate_vars)]

            model = sm.OLS(endog, new_exog).fit()
            model_loss = loss_fn(model)
            if model_loss < best_model_run_loss:
                best_model_run = model
                best_model_run_loss = model_loss

        if best_model_run_loss < best_model_loss:
            best_model = best_model_run
            best_model_loss = best_model_run_loss
            remaining_vars = set(best_model.params.index)
        else:
            return best_model


def main():
    endog, exog = load_car_mileage_data()
    mallows_cp = create_mallows_cp(endog, exog)

    fs_model = forward_stepwise(endog, exog, mallows_cp)
    fs_params = set(fs_model.params.index)
    fs_model_loss = mallows_cp(fs_model)
    print("Best forward stepwise model covariates:", *fs_params)
    print(f"Best forward stepwise model Mallow Cp: {fs_model_loss:.2f}")

    bs_model = backward_stepwise(endog, exog, mallows_cp)
    bs_params = set(bs_model.params.index)
    bs_model_loss = mallows_cp(bs_model)
    print("Best backward stepwise model covariates:", *bs_params)
    print(f"Best backward stepwise model Mallow Cp: {bs_model_loss:.2f}")


if __name__ == "__main__":
    main()
