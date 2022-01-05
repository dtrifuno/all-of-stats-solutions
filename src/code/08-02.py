import numpy as np

from bootstrap import (
    between,
    create_normal_ci,
    create_percentile_ci,
    create_pivotal_ci,
)

rng = np.random.default_rng(42)

b = 500
n = 50
sims = 1000

skewness = (
    lambda data: np.sum(((data - data.mean()) / data.std()) ** 3) / data.size
)
true_t = (np.e + 2) * np.sqrt(np.e - 1)

ci_coverage_counts = np.zeros(3)

for _ in range(sims):
    ys = rng.normal(size=n)
    xs = np.exp(ys)
    t_hat = skewness(xs)

    bootstrap_data = rng.choice(ys, size=(n, b), replace=True)
    bootstrap_ts = np.apply_along_axis(skewness, 0, bootstrap_data)

    normal_ci = create_normal_ci(t_hat, bootstrap_ts)
    ci_coverage_counts[0] += 1 if between(true_t, normal_ci) else 0
    pivotal_ci = create_pivotal_ci(t_hat, bootstrap_ts)
    ci_coverage_counts[1] += 1 if between(true_t, pivotal_ci) else 0
    percent_ci = create_percentile_ci(bootstrap_ts)
    ci_coverage_counts[2] += 1 if between(true_t, percent_ci) else 0

ci_coverages = ci_coverage_counts / sims
print("Estimated coverages of 95% CIs: ")
ci_coverages = ci_coverage_counts / sims
print(f"    Normal: {ci_coverages[0]:.3}")
print(f"   Pivotal: {ci_coverages[1]:.3}")
print(f"Percentile: {ci_coverages[2]:.3}")
