import numpy as np
import matplotlib.pyplot as plt


mean = 0
sigma = 15

x = np.linspace(-100, 100, 100)


pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mean)**2) / (2 * sigma**2))

# Plot
plt.plot(x, pdf)
plt.xlabel("x")
plt.ylabel("Gaussian PDF")
plt.title("Gaussian PDF (mean = 0, sigma = 15)")
plt.show()
