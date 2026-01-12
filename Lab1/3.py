import numpy as np
import matplotlib.pyplot as plt

# Generate x values
x = np.linspace(-10, 10, 100)

# Define the equation
y = 2*x**2 + 3*x + 4

# Plot
plt.plot(x, y, label='y = 2x² + 3x + 4')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of y = 2x² + 3x + 4')
plt.legend()
plt.grid(True)
plt.show()
