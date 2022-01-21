import statsmodels.api as sm

from ex13_06a import load_car_mileage_data

endog, exog = load_car_mileage_data()

full_model = sm.OLS(endog, exog)
res = full_model.fit()
print(res.summary())
