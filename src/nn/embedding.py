try:
    import cupy as np
except:
    import numpy as np
from .base import Layer

class Embedding(Layer):
    def __init__(self, input_d, output_d):
        self.input_d = input_d
        self.output_d = output_d
        self.weights = 0.01*np.random.randn(input_d, output_d)
        
    def onehot(self, inputs):
        self.batch_size, self.seq_len = len(inputs), len(inputs[0])
        inputs = inputs.astype(np.int32)
        
        batch_labels = np.zeros((inputs.size, self.input_d))
        batch_labels[np.arange(inputs.size), inputs.reshape(1, -1)] = 1

        return batch_labels.reshape(self.batch_size, self.seq_len, self.input_d) 
    
    def forward(self, inputs, training=True):
        self.inputs = self.onehot(inputs)
        self.outputs = np.matmul(self.inputs, self.weights)
        return self.outputs
    
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)
    
    def backward(self, dvalues):
        self.dweights = np.matmul(self.inputs.reshape(-1, self.input_d).T, dvalues.reshape(-1, self.output_d))
        self.dinputs = np.matmul(dvalues, self.weights.T)
        return self.dinputs
    
    @property
    def params(self):
        return [self.weights]
    @property
    def grads(self):
        return [self.dweights]
    
    