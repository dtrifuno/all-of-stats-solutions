import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

from ex13_06a import load_car_mileage_data

endog, exog = load_car_mileage_data()
endog["LOG_MPG"] = np.log(endog["MPG"])
endog = endog[["LOG_MPG"]]
exog = exog[["const", "HP"]]

mod = sm.OLS(endog, exog)
res = mod.fit()
print(res.summary())

fig, ax = plt.subplots(figsize=(7, 5))
xs = np.linspace(exog["HP"].min(), exog["HP"].max(), 100)
ys = res.predict(sm.add_constant(xs))
ax.set(xlabel="HP", ylabel="log(MPG)")
ax.scatter(exog["HP"], endog, alpha=0.5, label="Data")
ax.plot(xs, ys, color="tab:red", label="Model")
ax.legend()

fig.savefig("13-06b.png", bbox_inches="tight")
