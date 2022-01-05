import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 1, 40)
y = np.repeat(1, 40)
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("Density")
plt.savefig("03-19a.png", bbox_inches="tight")
