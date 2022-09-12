import numpy as np
import random

def flip_sequence(p=0.5):
    
    assert p >= 0, "p must be larger than or equal to zero"
    assert p <= 1, "p must be less than or equal to 1"

    def _augment(sequence,locations):
        if (random.random() > p):
            return sequence,locations
        else:   
            return np.negative(sequence), locations

    return _augment
