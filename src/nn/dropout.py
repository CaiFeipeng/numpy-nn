try:
    import cupy as np
except:
    import numpy as np
from .base import Layer

class Dropout(Layer):
    def __init__(self, rate=0.1) -> None:
        self.rate = rate
        self.scale = 1.0 / (1.0 - self.rate)
        
    def forward(self, x, training=True):
        
        self.mask = 1.0
        
        if training:
            self.mask = np.random.binomial(
                n = 1,
                p = 1 - self.rate,
                size = x.shape
            )
        return x * self.mask * self.scale
    
    def __call__(self, x, training=True, *args, **kwds):
        return self.forward(x, training=training, *args, **kwds)

    def backward(self, dvalues):
        self.dinputs = dvalues * self.mask * self.scale
        return self.dinputs
        