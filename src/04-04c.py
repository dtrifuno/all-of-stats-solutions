import matplotlib.pyplot as plt
import numpy as np


def length_of_interval(n, alpha=0.05):
    return 2 * np.sqrt((1 / (2 * n)) * np.log(2 / alpha))


ns = np.arange(60, 5000)
ys = np.vectorize(length_of_interval)(ns)

for n, y in zip(ns, ys):
    if y <= 0.05:
        print(f"Smallest n with interval width <= 0.05: {n}")
        break

plt.plot(ns, ys)
plt.xlabel("n")
plt.ylabel("Interval width")
plt.show()
