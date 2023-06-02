import numpy as np
from keras.datasets import mnist

import tsne
import graph

n_samples = 1000 # number of points
dim_low = 2 # embedding space dimension
d = 4 # number of work nodes

max_iter = 1000  # number of gradient descent steps
n_steps = 100  # number of binary search steps for sigma value

early_exaggeration = 12.0
desired_perplexity = 30.0
initial_momentum = 0.5
final_momentum = 0.8
eta = 50 # learning_rate
min_gain = 0.01

(train_X, train_y), (test_X, test_y) = mnist.load_data()
(k, m_coord1, m_coord2) = train_X.shape
m_coord = m_coord1 * m_coord2 # original space dimension
X = np.array(train_X.reshape((k, m_coord)).T[:, 0:n_samples], dtype=float)
y = np.array(train_y[0:n_samples])

# the original vector in a lower dimension space
Y = 1e-4 * np.random.standard_normal(size=(dim_low, n_samples)).astype(np.float32)

Y = tsne.tsne(X, Y, n_samples, m_coord, dim_low, d, max_iter, n_steps, early_exaggeration, desired_perplexity, eta)
graph.graph_fun(Y, y)
