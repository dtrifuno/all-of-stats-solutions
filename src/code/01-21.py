import matplotlib.pyplot as plt
import numpy as np


def generate_coin_flips(size=1000, p_heads=0.5, seed=42):
    rng = np.random.default_rng(seed)
    p_tails = 1 - p_heads
    return rng.choice(("H", "T"), size=size, p=(p_heads, p_tails))


def plot_coin_flips(number_of_flips, p_heads):
    fig, ax = plt.subplots()

    flips = generate_coin_flips(number_of_flips, p_heads)
    proportion = np.cumsum(flips == "H") / np.arange(1, number_of_flips + 1)
    xs = np.arange(1, number_of_flips + 1)

    ax.plot(xs, proportion, linewidth=1.2)
    ax.set(xlabel="Number of coin tosses", ylabel="Proportion of heads")
    return fig


plot_coin_flips(1000, 0.3).savefig("01-21a.png", bbox_inches="tight")
plot_coin_flips(1000, 0.03).savefig("01-21b.png", bbox_inches="tight")
