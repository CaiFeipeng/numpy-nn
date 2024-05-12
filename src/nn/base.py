
from typing import Any


class Layer:
    def forward(self, *args, **kwargs):
        """forward"""
        raise NotImplemented
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(**args, **kwds)
    
    def backward(self, *args, **kwargs):
        """backward"""
        raise NotImplemented
    
    @property
    def params(self):
        return []
    
    @property
    def grads(self):
        return []
    
    @property
    def param_grads(self):
        return list(zip(self.params, self.grads))
    
    def __str__(self) -> str:
        return self.__class__.__name__