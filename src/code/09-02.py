import numpy as np

rng = np.random.default_rng(42)

n = 10
a = 1
b = 3
samples = 30000

tau_func = lambda data: ((data.max() + data.min()) / 2 - (a + b) / 2) ** 2
data = rng.uniform(a, b, size=(n, samples))
tau_mse = np.apply_along_axis(tau_func, 0, data)

print(f"The MSE for the MLE estimate is {tau_mse.mean():.3}.")
print(f"The MSE for the plug-in estimator is {(a + b) / (12 * n):.3}.")
