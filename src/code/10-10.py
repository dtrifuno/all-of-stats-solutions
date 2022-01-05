import numpy as np
import pandas as pd
import scipy.stats as stats

df = pd.DataFrame(
    data={"chinese": [55, 33, 70, 49], "jewish": [141, 145, 139, 161]},
    index=[-2, -1, 1, 2],
)
chinese_n = df["chinese"].sum()
chinese_p_hat = df["chinese"][:2].sum() / chinese_n
chinese_se_hat = np.sqrt(chinese_p_hat * (1 - chinese_p_hat) / chinese_n)

jewish_n = df["jewish"].sum()
jewish_p_hat = df["jewish"][:2].sum() / jewish_n
jewish_se_hat = np.sqrt(jewish_p_hat * (1 - jewish_p_hat) / jewish_n)

diff_p_hat = chinese_p_hat - jewish_p_hat
diff_se_hat = np.sqrt(chinese_se_hat ** 2 + jewish_se_hat ** 2)
wald = np.abs(diff_p_hat) / diff_se_hat
p = 2 * stats.norm.cdf(-wald)
print(f"The p-value from the Wald test is {p:.4}.")
