import numpy as np

_mask = None

def add_mask(arr):
    global _mask
    _mask = np.random.randn(*arr.shape)
    return arr + _mask

def remove_mask(arr):
    global _mask
    return arr - _mask if _mask is not None else arr
