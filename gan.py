from gan_dense import GanDense
import tensorflow as tf
import numpy as np
from PCA import load_mnist
import matplotlib.pyplot as plt

NOISE_DIM = 100
BATCH_SIZE = 256
EPOCHS = 100

X = load_mnist()
model = GanDense(X, BATCH_SIZE)

model.fit(epochs=EPOCHS, noise_dim=NOISE_DIM)