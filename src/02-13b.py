import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(42)
x = rng.normal(size=10000)
y = np.exp(x)
plt.xlabel("y", size=14)
plt.ylabel("Frequency", size=14)
plt.hist(y, bins=np.linspace(0, 6, 30))
plt.show()
