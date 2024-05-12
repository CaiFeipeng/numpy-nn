import src.nn as nn

class Sequential(nn.Layer):
    def __init__(self, layers) -> None:
        self.layers = layers
        
    def add(self, layer):
        self.layers.append(layer)
        
    def forward(self, x, training=True):
        for layer in self.layers:
            x = layer(x, training)
        return x
    
    def backward(self, dvalues):
        for idx, layer in enumerate(reversed(self.layers)):
            dvalues = layer.backward(dvalues)
        return dvalues
            
    
    def __call__(self, x, training=True, *args, **kwds):
        return self.forward(x, training, *args, **kwds)
    
    @property
    def params(self):
        params = []
        for layer in reversed(self.layers):
            params += layer.params
        return params
    @property
    def grads(self):
        grads = []
        for layer in reversed(self.layers):
            grads += layer.grads
        return grads