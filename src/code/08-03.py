import numpy as np
import scipy.stats as stats

from bootstrap import (
    between,
    create_normal_ci,
    create_percentile_ci,
    create_pivotal_ci,
)

rng = np.random.default_rng(42)

n = 25
sims = 100  # number of simulations
b = 5000  # number of bootstrap samples

ci_coverage_counts = np.zeros(3)
ci_lengths = np.zeros((3, sims))

for m in range(sims):
    data = stats.t.rvs(3, size=n, random_state=rng)
    true_q75 = stats.t.ppf(0.75, df=3)
    true_q25 = stats.t.ppf(0.25, df=3)
    true_theta = (true_q75 - true_q25) / 1.34

    est_q75 = np.quantile(data, 0.75)
    est_q25 = np.quantile(data, 0.25)
    est_theta = (est_q75 - est_q25) / 1.34

    coverage_counts = [0, 0, 0]
    bootstrap_data = rng.choice(data, size=(n, b), replace=True)
    q75 = np.quantile(bootstrap_data, 0.75, axis=0)
    q25 = np.quantile(bootstrap_data, 0.25, axis=0)
    bootstrap_thetas = (q75 - q25) / 1.34

    normal_ci = create_normal_ci(est_theta, bootstrap_thetas)
    ci_lengths[0, m] = normal_ci[1] - normal_ci[0]
    ci_coverage_counts[0] += 1 if between(true_theta, normal_ci) else 0

    pivotal_ci = create_pivotal_ci(est_theta, bootstrap_thetas)
    ci_lengths[1, m] = pivotal_ci[1] - pivotal_ci[0]
    ci_coverage_counts[1] += 1 if between(true_theta, pivotal_ci) else 0

    percent_ci = create_percentile_ci(bootstrap_thetas)
    ci_lengths[2, m] = percent_ci[1] - percent_ci[0]
    ci_coverage_counts[2] += 1 if between(true_theta, percent_ci) else 0


mean_ci_lengths = ci_lengths.mean(axis=1)
print("Average length of 95% CIs: ")
print(f"    Normal: {mean_ci_lengths[0]:.4}")
print(f"   Pivotal: {mean_ci_lengths[1]:.4}")
print(f"Percentile: {mean_ci_lengths[2]:.4}")
print()

print("Coverages of 95% CIs: ")
ci_coverages = ci_coverage_counts / sims
print(f"    Normal: {ci_coverages[0]:.3}")
print(f"   Pivotal: {ci_coverages[1]:.3}")
print(f"Percentile: {ci_coverages[2]:.3}")
