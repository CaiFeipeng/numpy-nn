try:
    import cupy as np
except:
    import numpy as np

class Optimizer:
    def __init__(self, layers) -> None:
        self.layers = layers
    
    def update(self):
        self.pre_update()
        for layer in self.layers:
            self.update_param(layer)
        self.post_update()
            
    def pre_update(self):
        pass
    def update_param(self):
        pass
    def post_update(self):
        pass    
    
class SGDOptimizer(Optimizer):
    def __init__(self, model, learning_rate=1., decay=0, momentum=0) -> None:
        self.layers = model
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.iters = 0
        
    def pre_update(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay*self.iters))
            
    def update_param(self, layer):
        params, grads = layer.params, layer.grads
        
        if self.momentum and not hasattr(layer, 'momentums'):
            layer.momentums = [np.zeros_like(p) for p in params]
        
        for idx, (param, grad) in enumerate(zip(params, grads)):
            if self.momentum:
                param_updates = self.momentum * layer.momentums[idx] - self.current_learning_rate * grad
            else:
                param_updates = -self.current_learning_rate * grad
            
            param += param_updates
    
    def pose_update(self):
        self.iters += 1