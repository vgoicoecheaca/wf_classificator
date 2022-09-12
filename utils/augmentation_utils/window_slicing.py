import tensorflow as tf
import random
import numpy as np


def window_slicing(p=0.5):
    assert p >= 0, "p must be larger than or equal to zero"
    assert p <= 1, "p must be less than or equal to 1"

    def _augment(sequence,locations):
        if random.random()>p:
            return sequence,locations       

        max_samples_rolling = 500
        rolling_length = random.randint(0,len(sequence))
        #rolling_length = random.randint(0,max_samples_rolling)
        sequence       = np.roll(sequence,rolling_length)
        #locations      = tf.math.add(locations,rolling_length)
        locations      = tf.math.floormod(tf.math.add(tf.cast(locations,dtype=tf.float32),rolling_length),tf.constant(sequence.shape[0],dtype=tf.float32))
        return sequence,locations
    
    return _augment
