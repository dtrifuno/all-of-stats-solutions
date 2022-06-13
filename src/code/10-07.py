import numpy as np
import scipy.stats as stats

np.set_printoptions(precision=3)

twain_data_str = ".225 .262 .217 .240 .230 .229 .235 .217"
snodgrass_data_str = ".209 .205 .196 .210 .202 .207 .224 .223 .220 .201"
twain_data = np.array([float(x) for x in twain_data_str.split()])
snodgrass_data = np.array([float(x) for x in snodgrass_data_str.split()])

# Wald test
twain_mu = twain_data.mean()
snodgrass_mu = snodgrass_data.mean()
diff_hat = twain_mu - snodgrass_mu
diff_se = np.sqrt(
    twain_data.var() / twain_data.size
    + snodgrass_data.var() / snodgrass_data.size
)

alpha = 0.05
z_alpha = stats.norm.ppf(1 - alpha / 2)
ci = diff_hat + z_alpha * diff_se * np.array([-1, 1])
w = np.abs(diff_hat - 0) / diff_se
p_value = 2 * stats.norm.cdf(-w)
print(f"A 95% CI for the difference of means is given by {ci}.")
print(f"The p-value for the Wald test is {p_value:.5f}.")


# permutation test
""" 
The combined sample size is too large to iterate over all permutations,
so we sample instead.
"""
rng = np.random.default_rng()
joined_data = np.concatenate((twain_data, snodgrass_data))
twain_size = twain_data.size
b = 200000
positive_perms = 0
for _ in range(b):
    permuted_data = np.random.permutation(joined_data)
    perm_twain_data = permuted_data[:twain_size]
    perm_snodgrass_data = permuted_data[twain_size:]
    perm_diff = perm_twain_data.mean() - perm_snodgrass_data.mean()
    if perm_diff > diff_hat:
        positive_perms += 1

perm_p_value = positive_perms / b
print(f"The p-value for the permutation test is {perm_p_value:.5}.")
