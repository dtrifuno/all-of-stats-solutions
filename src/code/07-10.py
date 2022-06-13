import pathlib

import numpy as np
import pandas as pd
import scipy.stats as stats

np.set_printoptions(precision=3)

src_dir = pathlib.Path(__file__).resolve().parent.parent
dat_path = src_dir.joinpath("data", "clouds.dat")

df = pd.read_table(dat_path, sep="\s+", skiprows=29)
unseeded_rainfall, seeded_rainfall = df["Unseeded_Clouds"], df["Seeded_Clouds"]

alpha = 0.05
z_alpha = stats.norm.ppf(1 - alpha / 2)

unseeded_mean = unseeded_rainfall.mean()
unseeded_mean_se = unseeded_rainfall.std() / np.sqrt(unseeded_rainfall.size)
seeded_mean = seeded_rainfall.mean()
seeded_mean_se = seeded_rainfall.std() / np.sqrt(seeded_rainfall.size)

diff = seeded_mean - unseeded_mean
diff_se = np.sqrt(unseeded_mean_se ** 2 + seeded_mean_se ** 2)
diff_ci = diff + z_alpha * diff_se * np.array([-1, 1])

print(f"The mean waiting time is approximately {diff:.4}.")
print(f"A 95% CI for this value is given by {diff_ci}.")
