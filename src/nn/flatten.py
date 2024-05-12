try:
    import cupy as np
except:
    import numpy as np
from .base import Layer

class Flatten(Layer):
    def __init__(self):
        self.shape = None
        
    def forward(self, inputs, training=True):
        self.shape = inputs.shape
        return inputs.reshape(self.shape[0], -1)
    
    def __call__(self, inputs, *args, **kwds):
        return self.forward(inputs, *args, **kwds)
    
    def backward(self, dvalues):
        return dvalues.reshape(self.shape)
        