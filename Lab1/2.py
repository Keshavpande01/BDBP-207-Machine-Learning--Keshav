import matplotlib.pyplot as plt

# Parameters
start = -100
stop = 100
num = 100

# Generate x1 values
step = (stop - start) / (num - 1)
x1 = []
for i in range(num):
    x1.append(start + i * step)

# Linear equation y = 2x1 + 3
y = []
for x in x1:
    y.append(2 * x + 3)

# Plot
plt.plot(x1, y)
plt.xlabel("x1")
plt.ylabel("y")
plt.title("y = 2x1 + 3")
plt.show()
