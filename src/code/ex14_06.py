import numpy as np

from ex14_04 import generate_multivariate_normal
from ex14_05 import get_fisher_rho_ci, get_parametric_rho_ci

rng = np.random.default_rng(42)

mean = np.array([3, 8])
cov = np.array([[1, 1], [1, 2]])
corr = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

nsims = 1000
coverages = np.zeros(2)
for _ in range(nsims):
    samples = generate_multivariate_normal(100, mean, cov, rng)
    fisher_ci = get_fisher_rho_ci(samples)
    parametric_ci = get_parametric_rho_ci(samples, nsims=250, rng=rng)

    if (corr >= fisher_ci[0]) and (corr <= fisher_ci[1]):
        coverages[0] += 1

    if (corr >= parametric_ci[0]) and (corr <= parametric_ci[1]):
        coverages[1] += 1

coverages = coverages / nsims
print("Coverage of the 95% Fisher method CI for rho:", f"{coverages[0]:.3f}")
print(
    "Coverage of the 95% parametric bootstrap CI for rho:",
    f"{coverages[1]:.3f}",
)
