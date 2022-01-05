import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(2, 2, figsize=(14, 12))
rng = np.random.default_rng(42)
for k, n in enumerate((1, 5, 25, 100)):
    print(f"n: {n}")
    samples = rng.uniform(size=(n, 300_000))
    sample_means = np.average(samples, axis=0)

    i, j = k // 2, k % 2
    axs[i, j].hist(sample_means, bins="auto", range=(0, 1))
    axs[i, j].set(xlabel="Sample mean", ylabel="Frequency")
    axs[i, j].set_title(f"n = {n}")

    expected_mean = 1 / 2
    sample_mean = np.average(sample_means)
    print(f"expected average of sample mean: {expected_mean:.4}")
    print(f"average sample mean: {sample_mean:.4}")

    expected_var = 1 / (12 * n)
    sample_var = np.var(sample_means)
    print(f"expected variance of sample mean: {expected_var:.4}")
    print(f"sample variance of sample mean: {sample_var:.4}")
    print()

fig.savefig("03-19d.png", bbox_inches="tight")
