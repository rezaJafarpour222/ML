# Seed for reproducibility
from matplotlib import pyplot as plt
import numpy as np
from pandas import set_option

from myPerceptron import Perceptron
from utils import plotter


np.random.seed(42)

# Number of samples per class
n_samples = 50

# Class +1 centered at (2, 2)
X_pos = np.random.randn(n_samples, 2) + 2
Y_pos = np.ones(n_samples)

# Class -1 centered at (-2, -2)
X_neg = np.random.randn(n_samples, 2) - 2
Y_neg = -np.ones(n_samples)

# Combine the data
X = np.vstack((X_pos, X_neg))
Y = np.concatenate((Y_pos, Y_neg))

# Shuffle the dataset
indices = np.random.permutation(len(Y))
X = X[indices]
Y = Y[indices]

# Split into train/test
split = int(0.8 * len(Y))
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]
model = Perceptron(X.shape[1])
model.SGD(
    X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test, lr=0.001, epochs=1
)
plt.scatter(X_pos[:, 0], X_pos[:, 1], marker="o", label="Class +1")
plt.scatter(X_neg[:, 0], X_neg[:, 1], marker="x", label="Class -1")
w1, w2 = model.W
x_vals = np.array([X_train[:, 0].min() - 1, X_train[:, 0].max() + 1])
y_vals = -(w1 / w2) * x_vals
plt.plot(x_vals, y_vals, label="Decision Boundary")

plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.title("Perceptron Decision Boundary")
plt.show()
