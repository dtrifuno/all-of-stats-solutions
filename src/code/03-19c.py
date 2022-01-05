import matplotlib.pyplot as plt
import numpy as np

n = np.arange(1, 101)
y = 1 / (12 * n)
plt.plot(n, y)
plt.xlabel("n")
plt.ylabel("Variance of sample mean")
plt.savefig("03-19c.png", bbox_inches="tight")
