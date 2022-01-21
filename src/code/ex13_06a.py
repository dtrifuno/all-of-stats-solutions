import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_car_mileage_data():
    src_dir = pathlib.Path(__file__).resolve().parent.parent
    dat_path = src_dir.joinpath("data", "carmileage.dat")
    df = pd.read_table(
        dat_path,
        sep="\s+",
        skiprows=28,
        names=["MAKE/MODEL", "VOL", "HP", "MPG", "SP", "WT"],
    )

    endog = df[["MPG"]]
    exog = sm.add_constant(df[["VOL", "HP", "SP", "WT"]])
    return endog, exog


def main():
    endog, exog = load_car_mileage_data()
    exog = exog[["const", "HP"]]

    mod = sm.OLS(endog, exog)
    res = mod.fit()
    print(res.summary())

    fig, ax = plt.subplots(figsize=(7, 5))
    xs = np.linspace(exog["HP"].min(), exog["HP"].max(), 100)
    ys = res.predict(sm.add_constant(xs))
    ax.set(xlabel="HP", ylabel="MPG")
    ax.scatter(exog["HP"], endog, alpha=0.5, label="Data")
    ax.plot(xs, ys, color="tab:red", label="Model")
    ax.legend()

    fig.savefig("13-06a.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
