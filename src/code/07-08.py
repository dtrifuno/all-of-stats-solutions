import pathlib

import numpy as np
import pandas as pd
import scipy.stats as stats

np.set_printoptions(precision=3)

src_dir = pathlib.Path(__file__).resolve().parent.parent
dat_path = src_dir.joinpath("data", "faithful.dat")

df = pd.read_table(dat_path, sep="\s+", index_col=0, skiprows=25)
waiting_times = df["waiting"]

mean_waiting_time = waiting_times.mean()
print(f"The mean waiting time is approximately {mean_waiting_time:.4}.")

mwt_se = waiting_times.std() / np.sqrt(waiting_times.size)
print(f"The standard error of this estimate is {mwt_se:.4}.")
alpha = 0.1
z_alpha = stats.norm.ppf(1 - alpha / 2)
mwt_ci = mean_waiting_time + z_alpha * mwt_se * np.array([-1, 1])
print(f"A 90% CI for the mean waiting time value is given by {mwt_ci}.")

median_waiting_time = waiting_times.median()
print(f"The median waiting time is approximately {median_waiting_time:.4}.")
