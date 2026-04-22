import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [1,13], [1,18], [2,9], [3,6], [6,3], [9,2], [13,1], [18,1],  # Label - Blue
    [3,15], [6,6], [6,11], [9,5], [10,10], [11,5], [12,6], [16,3]   # Label - Red
])

# 2-D plot of the above original points -
plt.figure(figsize=(8,6))    # width: 8 inches, height: 6 inches
plt.scatter(X[:8, 0], X[:8, 1], c='blue', label='Blue')   # First eight are blue; selects the x1 ([:8, 0]) , x2 coordinates ([:8, 1])
plt.scatter(X[8:, 0], X[8:, 1], c='red', label='Red')   # Last eight are red
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Original 2D plot')
plt.legend()
plt.show()

# Transform these points into 3-D space -
def Transform(x):     # Feature mapping function (takes a data point 'x' as input)
    x1, x2 = x[0], x[1]    # Extracts the x1 & x2 coordinates from the input point
    return np.array([x1**2, np.sqrt(2)*x1*x2, x2**2])     # returns a new array representing the transformed 3D point

# Transform all the points -
X_transformed = np.array([Transform(x) for x in X])    # applies the 'Transform' function to each point in the 'X' array & converts the result back into an array.

# 3-D plot of transformed points -
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')    # Adds a 3D subplot to the figure, '111' means 1 row, 1 column, first subplot, projection='3d' specifies that it should be a 3D plot.
ax.scatter(X_transformed[:8, 0], X_transformed[:8, 1], X_transformed[:8, 2], c='blue', label='Blue')     # Creates a 3D scatter plot of the first eight transformed data points (Blue)
ax.scatter(X_transformed[8:, 0], X_transformed[8:, 1], X_transformed[8:, 2], c='red', label='Red')    # Creates a 3D scatter plot of the last eight transformed data points (Red)
ax.set_xlabel('x1^2')
ax.set_ylabel('sqrt(2)*x1*x2')
ax.set_zlabel('x2^2')
ax.set_title('Transformed 3D plot')
ax.legend()
plt.show()

# Ques-1
x1 = np.array([3,6])
x2 = np.array([10,10])
x1_transformed = Transform(x1)
x2_transformed = Transform(x2)
dot_product = np.sum(x1_transformed * x2_transformed)    # Calculates the dot product of the transformed data points 'x1_transformed' and 'x2_transformed'.
print(f"Dot product in a higher dimension: {dot_product}")

# Ques-2
def K(a, b):    # takes 2 data points 'a' and 'b' as input & calculates the polynomial kernel
    return a[0]**2*b[0]**2+2*a[0]*b[0]*a[1]*b[1]+a[1]**2*b[1]**2
kernel_output = K(x1, x2)
print(f"Polynomial kernel result: {kernel_output}")

# The dot product in the higher dimension & the polynomial kernel result are same bec. the later one is designed to compute the dot product in that higher-dimensional space without explicitly transforming the data.
# This is the core idea behind the "kernel trick". Its advantages are - computational efficiency, abstraction, and handling non-linear data.