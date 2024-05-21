try:
    import cupy as np
except:
    import numpy as np
from .base import Layer

class BatchNorm2D(Layer):
    # need to be done
    def __init__(self, input_d, momentum=0.9) -> None:
        self.gamma = np.ones(shape=(1, input_d, 1, 1))
        self.beta = np.zeros(shape=(1, input_d, 1, 1))
        
        self.running_mean = np.zeros(shape=(1, input_d, 1, 1))
        self.running_var = np.zeros(shape=(1, input_d, 1, 1))
        
        self.eps = 1e-10
        self.momentum = momentum
        
    def forward(self, inputs, training=True):
        self.inputs = inputs
        # for training only
        if training:
            mean = inputs.mean(axis=(0,2,3), keepdims=True)
            var = inputs.var(axis=(0,2,3), keepdims=True)

            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var
            
            self.std = np.sqrt(var + self.eps)
            self.inputs_centered = inputs - mean
            self.inputs_norm = self.inputs_centered / self.std

            out = self.gamma * self.inputs_norm + self.beta
        else:
            inputs_norm = (inputs - self.running_mean) / (np.sqrt(self.running_var + self.eps))
            out = self.gamma * inputs_norm + self.beta

        return out
    
    def __call__(self, inputs, training=True, *args, **kwds):
        return self.forward(inputs, training, *args, **kwds)
    
    def backward(self, dvalues):
        # dvalues: [batch_size, C, H, W]
        N, C, H, W = dvalues.shape
        self.dbeta = np.sum(dvalues, axis=(0,2,3), keepdims=True)
        self.dgamma = np.sum(dvalues * self.inputs_norm, axis=(0, 2, 3), keepdims=True)

        # calculate input gradients
        dinput_norm = dvalues * self.gamma
        dstd = np.sum(-1 * self.inputs_centered * self.std**(-2), axis=(0, 2, 3), keepdims=True)
        dinput_centered = dinput_norm / self.std
        
        dvar = 0.5 * 1 / self.std * dstd
        dsq = 1 / (N*H*W) * np.ones((N,C,H,W)) * dvar
        dinput_centered_var = 2 * self.inputs_centered * dsq
        
        dinput_centered = dinput_centered + dinput_centered_var
        
        dmu = -1 * np.sum(dinput_centered, axis=(0, 2, 3), keepdims=True)
        dx_mu = 1 / (N * H * W) * np.ones((N,C,H,W)) * dmu
        
        dinputs = dinput_centered + dx_mu
        return dinputs
    
    @property
    def params(self):
        return [self.gamma, self.beta]
    @property
    def grads(self):
        return [self.dgamma, self.dbeta]
        
        
        

class LayerNorm(Layer):
    def __init__(self, input_d):
        self.input_d = input_d
        self.gamma = np.ones(shape=(input_d))
        self.beta = np.zeros(shape=(input_d))
        self.eps = 1e-10
        
    def forward(self, inputs, training=True):
        self.inputs = inputs
        mean = inputs.mean(axis=(-1), keepdims=True)
        self.std = inputs.std(axis=(-1), keepdims=True) + self.eps
        self.inputs_centered = inputs - mean
        outputs = self.gamma * self.inputs_centered / (self.std) + self.beta
        return outputs
    
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)
    
    def backward(self, dvalues):
        dim = dvalues.shape[-1]
        
        self.dgamma = np.sum(dvalues * self.inputs_centered / self.std, axis=0, keepdims=True)
        self.dgamma = np.sum(self.dgamma.reshape(-1, dim), axis=0)
        self.dbeta = np.sum(dvalues.reshape(-1, dim), axis=0)
        
        dxhat = dvalues * self.gamma
        dinput_centered = dxhat / self.std
        dvar = -0.5 * np.sum(dxhat * self.inputs_centered * self.std **(-3), axis=-1, keepdims=True) 
        dsq = dvar * 1/dim
        dinput_centered_var = 2 * self.inputs_centered * dsq
        
        dinput_centered = dinput_centered + dinput_centered_var
        
        dmu = -1 * np.sum(dinput_centered, axis=-1, keepdims=True)
        dinput_mu = 1/dim * dmu
        dinputs = dinput_centered + dinput_mu
        # dlxhat = dvalues * self.gamma
        # dxhatx = 1 / self.std
        # dlvar = -0.5 * np.sum(self.gamma * self.inputs_mu * self.std**(-3) * dvalues, axis=-1, keepdims=True)
        # dvarx = 2 * self.inputs_mu / dim
        # dlmu = -1 * np.sum(dlxhat / self.std, axis=-1, keepdims=True) + -2 / dim * np.sum(dlvar * self.inputs_mu)
        
        # self.dinput = dlxhat * dxhatx + dlvar * dvarx + dlmu / dim
        
        return dinputs
        
    @property
    def params(self):
        return [self.gamma, self.beta]
    
    @property
    def grads(self):
        return [self.dgamma, self.dbeta]

class RMSNorm(Layer):
    pass

if __name__=='__main__':
    x = np.arange(96).reshape(2, 3, 4, 4)
    norm = BatchNorm2D(3)
    out = norm(x)
    print(out)
    pass