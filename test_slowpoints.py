# %load_ext autoreload
# %autoreload 2

import tensorflow as tf
import scipy
import numpy as np
from slowpoints import find_slow_points, visualize_slow_points
import pdb
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import json
import pandas as pd
from sklearn.decomposition import PCA
import pickle

with open('weight_history_0.json') as f:
    weight_history = json.load(f)

def RNN_RHS(x, W, bias, activation_fn, u=None, Wu=None):
    if u is None:
        u = np.zeros(x.shape)
        Wu = np.zeros(W.shape)

    return -1 * x + activation_fn(jnp.matmul(W, x) + jnp.matmul(Wu, u) + bias)

# Generate 500 initial conditions in the (-1, 1) hypercube
x0 = []
for i in range(500):
    x0.append(np.random.normal(-1, 1, size=(256,)))

W = np.array(weight_history['trained weights'][-1])
b = np.array(weight_history['bias'])

slow_points, J_eig = find_slow_points(lambda x: RNN_RHS(x, W, b, jax.nn.relu), x0)

with open("slow_points.txt", "wb") as fp:   #Pickling
    pickle.dump(slow_points, fp)
with open("J_eig.txt", "wb") as fp:   #Pickling
    pickle.dump(J_eig, fp)
