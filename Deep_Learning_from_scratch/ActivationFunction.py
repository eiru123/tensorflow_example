import numpy as np
import matplotlib.pyplot as plt


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
y2 = step_function(x)
plt.plot(x, y)
plt.plot(x, y2, linestyle="--")
plt.ylim(-0.1, 1.1)
# plt.show()

X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])
print(W1.shape)
print(X.shape)
print(B1.shape)

A1 = np.dot(X, W1) + B1
print(A1)

z = sigmoid(A1)
print(z)

W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

print(z.shape)
print(W2.shape)
print(B2.shape)

A2 = np.dot(z, W2) + B2

print(A2)
z2 = sigmoid(A2)
print(z2)


def identity_function(x):
    return x

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(z2, W3) + B3

Y = identity_function(A3)