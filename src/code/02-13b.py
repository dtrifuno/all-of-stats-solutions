import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(42)
x = rng.normal(size=10000)
y = np.exp(x)
plt.xlabel("y")
plt.ylabel("Frequency")
plt.hist(y, bins="rice", range=(0, 6))
plt.savefig("02-13b.png", bbox_inches="tight")
