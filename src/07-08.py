import numpy as np
import pandas as pd
import scipy.stats as stats

df = pd.read_table("src/data/faithful.dat", sep="\s+", index_col=0, skiprows=25)
waiting_times = df["waiting"]

mean_waiting_time = waiting_times.mean()
print(f"The mean waiting time is approximately {mean_waiting_time:.4}.")

mwt_se = waiting_times.std() / np.sqrt(waiting_times.size)
print(f"The standard error of this estimate is {mwt_se:.4}.")
alpha = 0.1
z = stats.norm.ppf(1 - alpha / 2)
mwt_lower_bound = mean_waiting_time - z * mwt_se
mwt_upper_bound = mean_waiting_time + z * mwt_se
print(
    "A 90% CI for the mean waiting time value is given by",
    f"[{mwt_lower_bound:.4}, {mwt_upper_bound:.4}].",
)

median_waiting_time = waiting_times.median()
print(f"The median waiting time is approximately {median_waiting_time:.4}.")
