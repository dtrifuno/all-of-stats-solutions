import matplotlib.pyplot as plt
import numpy as np

n = 10000
rng = np.random.default_rng(42)
x = rng.normal(size=n)
y = x.cumsum() / np.arange(1, n + 1)
plt.ylabel("Sample mean")
plt.xlabel("n")
plt.plot(y)
plt.show()
