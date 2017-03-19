import numpy as np
import tensorflow as tf


def xavier_weight_init():
    def _xavier_initializer(shape, **kwargs):
        epsilon = np.sqrt(6)/(shape[0]+shape[1])
        out = tf.random_uniform(shape, minval=-epsilon, maxval=epsilon, dtype=tf.float32)
        return out
    return _xavier_initializer