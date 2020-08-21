import numpy as np
import tensorflow as tf

class RBFExpansion:
    def __init__(self, low, high, count):
        self.low = low
        self.high = high
        self.centers = np.linspace(low, high, count).astype(np.float32)
        self.gap = self.centers[1] - self.centers[0]

    def __call__(self, d):
        # input shpe
        rbf = tf.map_fn(lambda x: tf.math.exp(-(x - self.centers)**2 / self.gap), tf.reshape(d, (-1,)))
        return tf.reshape(rbf, tf.concat((tf.shape(d), self.centers.shape), axis=0))
