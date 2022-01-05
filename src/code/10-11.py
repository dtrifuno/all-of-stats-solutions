import numpy as np
import pandas as pd
import scipy.stats as stats

index = [
    "Placebo",
    "Chlorpromazine",
    "Dimenhydrinate",
    "Pentobarbital (100 mg)",
    "Pentobarbital (150 mg)",
]
df = pd.DataFrame(
    data={"nausea": [45, 26, 52, 35, 37], "patients": [80, 75, 85, 67, 85]},
    index=index,
)
df["p_hat"] = df["nausea"] / df["patients"]
df["se_hat"] = np.sqrt(df["p_hat"] * (1 - df["p_hat"]) / df["patients"])

placebo = df.loc["Placebo"]
placebo_p_hat = placebo["p_hat"]
placebo_se_hat = placebo["se_hat"]

diff_se = np.sqrt(df["se_hat"] ** 2 + placebo_se_hat ** 2)
df["wald"] = np.abs(df["p_hat"] - placebo_p_hat) / diff_se
df["p"] = 2 * stats.norm.cdf(-df["wald"])
placebo_odds = placebo["nausea"] / (placebo["patients"] - placebo["nausea"])
df["odds"] = placebo_odds / (df["nausea"] / (df["patients"] - df["nausea"]))

# Bonferroni Method
df["bf_p"] = 4 * df["p"]

# Benjamini-Hochberg Method
alpha = 0.05
p_values = df[1:]["p"].to_numpy()
p_values.sort()
ls = np.arange(1, 5) * alpha / 4
r = np.argmax(np.cumsum(p_values < ls))
p_critical = p_values[r]
df["bh_reject"] = df["p"] <= p_critical

print(df.iloc[1:][["p_hat", "odds", "p", "bf_p", "bh_reject"]])
