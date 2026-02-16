

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    sig = sigmoid(z)
    return sig * (1 - sig)


z = np.linspace(-10, 10, 100)
y = sigmoid_derivative(z)

plt.figure(figsize=(10, 6))
plt.plot(z, y, label="Sigmoid Derivative")
plt.xlabel("z")
plt.ylabel("g'(z)")
plt.title("Derivative of Sigmoid Function")
plt.legend()
plt.grid(True)
plt.show()
