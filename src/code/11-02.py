import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# (a)
n = 100
rng = np.random.default_rng(42)

mu = 5
data = rng.normal(mu, 1, size=n)

# (b)
mu_lim = (3, 7)
k = 5000
mus = np.linspace(*mu_lim, k)
likelihood = np.ones(k)
for x in data:
    likelihood *= stats.norm.pdf(x, loc=mus)

mu_density = likelihood * 1
mu_density /= mu_density.sum()

fig_b, ax_b = plt.subplots()
ax_b.set(xlabel="Mu", ylabel="Posterior density")
ax_b.plot(mus, mu_density)
fig_b.savefig("11-02b.png", bbox_inches="tight")

# (c)
drawn_mus = rng.choice(mus, size=1000, p=mu_density)
fig_c, ax_c = plt.subplots()
ax_c.set(xlabel="Mu", ylabel="Frequency")
ax_c.hist(drawn_mus, bins=150, range=mu_lim)
fig_c.savefig("11-02c.png", bbox_inches="tight")

# (d)
drawn_thetas = np.exp(rng.choice(mus, size=10000, p=mu_density))

theta_lims = (75, 220)
fig_d, axs_d = plt.subplots(1, 2, figsize=(14, 6))
axs_d[0].set(xlabel="Theta", ylabel="Frequency")
axs_d[0].hist(drawn_thetas, bins="auto", range=theta_lims)

thetas = np.geomspace(*theta_lims, k)
log_thetas = np.log(thetas)
diff = np.tile(log_thetas, n).reshape((n, -1)).transpose() - data
propto = np.exp(-0.5 * np.sum(diff ** 2, axis=1))
theta_density = propto / propto.sum()
axs_d[1].set(xlabel="Theta", ylabel="Posterior density")
axs_d[1].plot(thetas, theta_density)

fig_d.savefig("11-02d.png", bbox_inches="tight")

# (e)
mu_cdf = mu_density.cumsum()
alpha = 0.05
mu_left_idx = np.argmax(mu_cdf >= alpha / 2)
mu_right_idx = np.argmax(mu_cdf >= 1 - alpha / 2)
mu_pi = (mus[mu_left_idx], mus[mu_right_idx])
print(
    "95% posterior interval for mu is given by",
    f"[{mu_pi[0]:.3f}, {mu_pi[1]:.3f}]",
)

# (f)
theta_cdf = theta_density.cumsum()
alpha = 0.05
theta_left_idx = np.argmax(theta_cdf >= alpha / 2)
theta_right_idx = np.argmax(theta_cdf >= 1 - alpha / 2)
theta_pi = (thetas[theta_left_idx], thetas[theta_right_idx])
print(
    "95% posterior interval for theta is given by",
    f"[{theta_pi[0]:.3f}, {theta_pi[1]:.3f}]",
)
