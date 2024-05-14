try:
    import cupy as np
except:
    import numpy as np
from .base import Layer

class ReLU(Layer):
    
    def forward(self, inputs, training=True):
        self.inputs = inputs
        outputs = np.maximum(0, inputs)
        return outputs
    
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)
    
    def backward(self, dvalues):
        self.dinputs  = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
        return self.dinputs
   
class LeakyReLU(Layer):
    def __init__(self, negative_slope=0.01) -> None:
        self.negative_slope=negative_slope
        
    def forward(self, inputs, training=True):
        self.inputs = inputs
        outputs = np.where(inputs <= 0, self.negative_slope * inputs, inputs)
        return outputs
    
    def backward(self, dvalues):
        dvalues = dvalues * np.where(self.inputs <= 0, self.negative_slope, 1)
        return dvalues
    
class Sigmoid(Layer):
    def forward(self, inputs, training=True):
        inputs = inputs - np.max(inputs, axis=-1, keepdims=True)
        neg_exp = np.exp(-inputs)
        self.sigmoid = 1. / (1 + neg_exp)
        return self.sigmoid
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)
    def backward(self, dvalues):
        self.dinputs =  dvalues * self.sigmoid * (1 - self.sigmoid)
        return self.dinputs
    
    
class SoftMax(Layer):
    def __init__(self):
        self.eps = 1e-10

    def forward(self, inputs, training=True):
        # self.inputs = inputs
        inputs = inputs - np.max(inputs, axis=-1, keepdims=True)
        self.exp = np.exp(inputs)
        self.softmax = self.exp / (self.exp.sum(axis = -1,keepdims=True) + self.eps)
        return self.softmax
    
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)
    
    def backward(self, dvalues):
        dim = self.softmax.shape[-1]
        
        exp_sum = self.exp.sum(axis=-1, keepdims=True) + self.eps
        dexp_1 = dvalues / exp_sum
        
        dexp_sum = np.sum(dvalues * self.exp, axis=-1, keepdims=True) * (-1) / exp_sum**(2)
        dexp_2 = np.repeat(dexp_sum, repeats=dim, axis=-1)
        
        dexp = dexp_1 + dexp_2
        self.dinputs = dexp * self.exp
        
        return self.dinputs
    