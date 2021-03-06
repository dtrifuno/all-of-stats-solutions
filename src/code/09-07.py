import numpy as np
import scipy.stats as stats

np.set_printoptions(precision=3)

n1 = 200
x1 = 160
p1_est = x1 / n1
n2 = 200
x2 = 148
p2_est = x2 / n2
psi_est = p1_est - p2_est
psi_se_delta = np.sqrt(p1_est * (1 - p1_est) / n1 + p2_est * (1 - p2_est) / n2)
alpha = 0.1
z_alpha = stats.norm.ppf(1 - alpha / 2)
print(f"The maximum likelihood estimate for psi is: {psi_est:.4}.")

psi_delta_ci = psi_est + z_alpha * psi_se_delta * np.array([-1, 1])
print(f"A 90% delta method CI for psi is: {psi_delta_ci}.")

b = 10000
rng = np.random.default_rng(42)
bootstrap_data = (
    rng.binomial(n1, p1_est, size=b) / n1
    - rng.binomial(n2, p2_est, size=b) / n2
)
psi_se_bootstrap = (bootstrap_data - psi_est).std()
psi_boostrap_ci = psi_est + z_alpha * psi_se_bootstrap * np.array([-1, 1])
print(f"A 90% bootstrap CI for psi is: {psi_boostrap_ci}.")
