import numpy as np

from bootstrap import create_normal_ci, create_percentile_ci, create_pivotal_ci


def sample_corr_plugin(ys, zs):
    ys_mean = ys.mean()
    zs_mean = zs.mean()
    numerator = np.sum((ys - ys_mean) * (zs - zs_mean))
    denominator = np.sqrt(
        np.sum((ys - ys_mean) ** 2) * np.sum((zs - zs_mean) ** 2)
    )
    return numerator / denominator


rng = np.random.default_rng(42)

b = 5000
lsat = np.array(
    [576, 635, 558, 578, 666, 580, 555, 661, 651, 605, 653, 575, 545, 572, 594]
)
gpa = np.array(
    [3.39, 3.30, 2.81, 3.03, 3.44, 3.07, 3.00, 3.43, 3.36, 3.13]
    + [3.12, 2.74, 2.76, 2.88, 2.95]
)
n = len(lsat)
idxs = np.arange(0, n)

est_corr = sample_corr_plugin(lsat, gpa)

bootstrap_corrs = np.zeros(b)
for i in range(b):
    random_idxs = rng.choice(idxs, size=n, replace=True)
    lsat_sample = lsat[random_idxs]
    gpa_sample = gpa[random_idxs]

    bootstrap_corrs[i] = sample_corr_plugin(lsat_sample, gpa_sample)

est_corr_se = bootstrap_corrs.std()

print(f"The plug-in estimate for the correlation is {est_corr:.5}.")
print(f"Bootstrap approximation for standard error is {est_corr_se:.4}.")
normal_ci = create_normal_ci(est_corr, bootstrap_corrs)
print(f"95% CI by normal method is [{normal_ci[0]:.4}, {normal_ci[1]:.4}]")
pivotal_ci = create_pivotal_ci(est_corr, bootstrap_corrs)
print(f"95% CI by pivotal method is [{pivotal_ci[0]:.4}, {pivotal_ci[1]:.4}]")
percent_ci = create_percentile_ci(bootstrap_corrs)
print(
    f"95% CI by percentile method is [{percent_ci[0]:.4}, {percent_ci[1]:.4}]"
)
