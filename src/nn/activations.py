try:
    import cupy as np
except:
    import numpy as np
from .base import Layer

class ReLU(Layer):
    
    def forward(self, inputs, training=True):
        self.inputs = inputs
        self.outputs = np.maximum(0, inputs)
        return self.outputs
    
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)
    
    def backward(self, dvalues):
        self.dinputs  = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
        return self.dinputs
    
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
        
        # expand_softmax = np.expand_dims(self.softmax, axis=-1)
        # repeat_softmax = np.repeat(expand_softmax, repeats=dim, axis=-1)
        # d_softmax = repeat_softmax * np.identity(self.softmax.shape[-1]) - np.matmul(expand_softmax, np.moveaxis(expand_softmax, -1,-2))
        # expand_dvalues = np.expand_dims(dvalues, axis=-2)
        # self.dinputs = np.matmul(expand_dvalues, d_softmax)
        # self.dinputs = np.squeeze(self.dinputs, axis=-2)
        return self.dinputs
    