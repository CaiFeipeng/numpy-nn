try:
    import cupy as np
except:
    import numpy as np
from .base import Layer

class Linear(Layer):
    def __init__(self, input_d, output_d, use_bias=True) -> None:
        self.input_d = input_d
        self.output_d = output_d
        self.use_bias = use_bias
        self.weights = 0.01*np.random.randn(input_d, output_d)
        if self.use_bias:
            self.biases = np.zeros((1, output_d))
        
    def forward(self, inputs, training=True):
        self.inputs = inputs
        
        if self.use_bias:
            self.outputs = np.matmul(inputs, self.weights) + self.biases
        else:
            self.outputs = np.matmul(inputs, self.weights)
        return self.outputs
    
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)
    
    def backward(self, dvalues):
        # self.inputs: b x in_d
        # dvalues: b x out_d
        # self.weights: in_d x out_d
        self.dweights = np.matmul(self.inputs.reshape(-1, self.input_d).T, dvalues.reshape(-1, self.output_d))
        # self.biases: 1 x out_d
        if self.use_bias:
            self.dbiases = np.sum(dvalues.reshape(-1, self.output_d), axis=0, keepdims=True)
        # backward dvalues
        # dvalues: b x out_d
        # self.weights: in_d x out_d
        # input_dvalues: 
        self.dinputs = np.dot(dvalues, self.weights.T)
        return self.dinputs
    
    @property
    def params(self):
        if self.use_bias:
            return [self.weights, self.biases]
        return [self.weights]
    
    @property
    def grads(self):
        if self.use_bias:
            return [self.dweights, self.dbiases]
        return [self.dweights]
    