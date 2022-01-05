import matplotlib.pyplot as plt
import numpy as np

p = np.array([0, 1 / 10, 1 / 10, 8 / 10])
cdf = p.cumsum()
x = np.array([0, 2, 3, 5, 7])

for x0, x1, y in zip(x, x[1:], cdf):
    plt.hlines(y, x0, x1)

plt.scatter(x[1:-1], cdf[:-1], color="C0", facecolor="white", s=12)
plt.scatter(x[1:-1], cdf[1:], color="C0", s=12)

plt.ylabel("Probability")
plt.xlabel("x")
plt.savefig("02-02.png", bbox_inches="tight")
