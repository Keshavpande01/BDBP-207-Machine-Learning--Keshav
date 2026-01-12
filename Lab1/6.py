import numpy as np

# Define the function
def y(x1, x2, x3):
    return 2*x1 + 3*x2 + 3*x3 + 4

# Gradient (constant for linear function)
gradient = np.array([2, 3, 3])

# Sample points
points = [
    (0, 0, 0),
    (1, 2, 3),
    (-1, 0, 2),
    (3, -2, 1)
]

print("Gradient of y at selected points:\n")

for p in points:
    value = y(*p)
    print(f"Point (x1, x2, x3) = {p}")
    print(f"y = {value}")
    print(f"Gradient ∇y = {gradient}\n")
