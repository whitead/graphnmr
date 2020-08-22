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
        x = tf.reshape(d, (-1,))
        rbf = tf.math.exp(-(x[:,tf.newaxis] - self.centers)**2 / self.gap)
        # remove 0s
        rbf *= tf.cast(x > 1e-5, tf.float32)[:,tf.newaxis]
        return tf.reshape(rbf, tf.concat((tf.shape(d), self.centers.shape), axis=0))
