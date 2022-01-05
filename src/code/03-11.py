import matplotlib.pyplot as plt
import numpy as np


def simulate_stock_run(n=10_000, seed=42):
    fig, ax = plt.subplots()

    rng = np.random.default_rng(seed)
    y = rng.choice([-1, 1], size=n)
    x = y.cumsum()
    ax.set(ylabel="Cumulative change in stock price")
    ax.set(xlabel="Day")
    ax.plot(x, linewidth=1.1)

    return fig


simulate_stock_run(seed=42).savefig("03-11a.png", bbox_inches="tight")
simulate_stock_run(seed=43).savefig("03-11b.png", bbox_inches="tight")
simulate_stock_run(seed=44).savefig("03-11c.png", bbox_inches="tight")
