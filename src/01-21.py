import numpy as np
import pandas as pd
import seaborn as sns


def generate_coin_flips(size=1000, p_heads=0.5, seed=42):
    rng = np.random.default_rng(seed)
    p_tails = 1 - p_heads
    return rng.choice(("H", "T"), size=size, p=(p_heads, p_tails))


def plot_coin_flips(number_of_flips, p_heads):
    flips = generate_coin_flips(number_of_flips, p_heads)
    proportion = np.cumsum(flips == "H") / np.arange(1, number_of_flips + 1)
    df = pd.DataFrame(
        {
            "proportion of heads": proportion,
            "number of flips": np.arange(1, number_of_flips + 1),
        }
    )
    sns.relplot(x="number of flips", y="proportion of heads", kind="line", data=df)


plot_coin_flips(1000, 0.3)
plot_coin_flips(1000, 0.03)
