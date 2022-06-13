import numpy as np
import scipy.stats as stats

from ex14_04 import generate_multivariate_normal


def get_sample_corr(xs, ys):
    mean_x = xs.mean()
    mean_y = ys.mean()
    s_x = xs.std()
    s_y = ys.std()
    n = xs.size
    return ((xs - mean_x) * (ys - mean_y)).sum() / ((n - 1) * s_x * s_y)


def get_fisher_rho_ci(samples, alpha=0.05):
    n = samples.shape[0]
    rho_hat = get_sample_corr(samples[:, 0], samples[:, 1])

    theta_hat = 0.5 * (np.log(1 + rho_hat) - np.log(1 - rho_hat))
    se_hat = 1 / np.sqrt(n - 3)
    z_alpha = stats.norm.ppf(1 - 0.5 * alpha)
    ab = theta_hat + np.array([-1, 1]) * z_alpha * se_hat
    return (np.exp(2 * ab) - 1) / (np.exp(2 * ab) + 1)


def get_parametric_rho_ci(
    samples, nsims=5_000, alpha=0.05, rng=np.random.default_rng()
):
    n = samples.shape[0]
    mean = samples.mean(axis=0)
    cov = np.cov(samples.T)

    bootstrap_rhos = np.zeros(nsims)
    for i in range(nsims):
        samples = generate_multivariate_normal(n, mean, cov)
        bootstrap_rhos[i] = get_sample_corr(samples[:, 0], samples[:, 1])
    return np.quantile(bootstrap_rhos, (0.5 * alpha, 1 - 0.5 * alpha))


def main():
    rng = np.random.default_rng(42)
    np.set_printoptions(precision=4)

    mean = np.array([3, 8])
    cov = np.array([[1, 1], [1, 2]])
    samples = generate_multivariate_normal(100, mean, cov, rng)

    corr = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    sample_mean = samples.mean(axis=0)
    print("Estimated mean: ")
    print(sample_mean, "\n")

    sample_cov = np.cov(samples.T)
    print("Estimated covariance matrix: ")
    print(sample_cov, "\n")

    sample_corr = get_sample_corr(samples[:, 0], samples[:, 1])
    print(f"True correlation: {corr:.4f}")
    print(f"Sample correlation: {sample_corr:.4f}")

    fisher_ci = get_fisher_rho_ci(samples)
    print(f"Fisher method 95% CI for rho: {fisher_ci}.")

    parametric_ci = get_parametric_rho_ci(samples, rng=rng)
    print(f"Parametric bootstrap 95% CI for rho: {parametric_ci}.")


if __name__ == "__main__":
    main()
