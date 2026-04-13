# Implement sigmoid function in python and visualize it.
# The sigmoid curve for the breast cancer dataset model was also visualized in the script

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
z = np.linspace(-10, 10, 100)
y = sigmoid(z)     # g(z)

plt.figure(figsize=(8, 6))
plt.plot(z, y, label='Sigmoid Function', color='blue')
plt.title("Logistic regression sigmoid curve")
plt.xlabel("z")
plt.ylabel("g(z)")
plt.show()