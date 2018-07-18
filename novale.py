import matplotlib.pyplot as plt
import numpy as np


x = range(100)
y = [val**2 for val in x]
plt.plot(x, y)
plt.show()

x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)
plt.plot(x, y)
plt.show()

x = np.linspace(-3, 2, 200)
Y = x ** 2 - 2 * x + 1.
plt.plot(x, Y)
plt.show()

# plotting multiple plots
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)
z = np.cos(x)
plt.plot(x, y)
plt.plot(x, z)
plt.show()

data1 = np.loadtxt('datasets\\scipy.txt')  # load the file
for val in data1.T:  # loop over each and every value in data1.T
    plt.plot(data1[:, 0], val)  # data1[:,0] is the first row in data1.T
plt.show()