try:
    import cupy as np
except:
    import numpy as np
from src.nn.dropout import Dropout
from .base import Layer

class PositionalEncoding(Layer):
    def __init__(self, max_len, output_d, dropout=0.1):
        self.output_d = output_d
        self.max_len = max_len
        self.dropout = Dropout(dropout)
        
        pe =np.zeros((max_len, output_d))
        position = np.arange(0, max_len)[:,None]
        div_term = np.exp(np.arange(0, output_d, 2) * (np.log(10000.0) / output_d))
        pe[:,0::2] = np.sin(position * div_term)
        pe[:,1::2] = np.cos(position * div_term)
        self.pe = pe[None, :, :]
        
    def forward(self, inputs, training=True):
        batch_size, seq_len, input_d = inputs.shape
        outputs = inputs + self.pe[:, :seq_len, :] # batch_size, seq_len, dim
        outputs = self.dropout(outputs, training=training)
        return outputs
    
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)
    
    def backward(self, dvalues):
        ddropput = self.dropout.backward(dvalues)
        self.dinputs = ddropput
        return self.dinputs
        