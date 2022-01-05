import matplotlib.pyplot as plt
import numpy as np

alpha = 0.05
ns = np.arange(60, 5000)
ys = 2 * np.sqrt((1 / (2 * ns)) * np.log(2 / alpha))
min_n = ns[np.argmax(ys <= 0.05)]
print(f"Smallest n with interval width <= 0.05: {min_n}")

plt.plot(ns, ys)
plt.xlabel("n")
plt.ylabel("Interval width")
plt.savefig("04-04c.png", bbox_inches="tight")
