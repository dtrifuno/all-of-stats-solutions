import numpy as np


def count_heads(number_of_flips, p_heads=0.5, seed=42):
    rng = np.random.default_rng(seed)
    p_tails = 1 - p_heads
    flips = rng.choice(("H", "T"), size=number_of_flips, p=(p_heads, p_tails))
    return np.count_nonzero(flips == "H")


p = 0.3
print("Flips  Expected  Actual")
for n in (10, 100, 1000):
    print(f"{n:5} {round(p * n):9} {count_heads(n, p):7}")
