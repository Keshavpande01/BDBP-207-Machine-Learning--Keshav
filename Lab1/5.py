import numpy as np
import matplotlib.pyplot as plt

# Range
x1 = np.linspace(-10, 10, 100)

# Function
y = x1**2

# Plot
plt.plot(x1, y)
plt.xlabel("x1")
plt.ylabel("y")
plt.title("y = x1^2")
plt.show()
