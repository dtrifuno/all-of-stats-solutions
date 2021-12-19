import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0.0000001, 6, 10000)
y = (1 / (x * np.sqrt(2 * np.pi))) * np.exp(-0.5 * np.log(x) ** 2)
plt.xlabel("y", size=14)
plt.ylabel("Density", size=14)
plt.plot(x, y)
plt.show()
