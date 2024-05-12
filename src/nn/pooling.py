try:
    import cupy as np
except:
    import numpy as np
from src.utils.im2col import im2col, col2im
from .base import Layer

class AvgPooling(Layer):
    def __init__(self, pool_h, pool_w, stride=2, padding=0) -> None:
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.padding = padding
        
    def forward(self, inputs, training=True):
        N, self.C_in, self.H_in, self.W_in = inputs.shape
        
        self.H_out = (self.H_in + 2*self.padding - self.pool_h) // self.stride + 1
        self.W_out = (self.W_in + 2*self.padding - self.pool_w) // self.stride + 1
        
        cols = im2col(inputs, self.pool_h, self.pool_w, padding=self.padding, stride=self.stride)
        cols = cols.reshape(N, self.C_in, self.pool_h*self.pool_w, self.H_out, self.W_out)
        avg_pools = np.mean(cols, axis = 2)
        return avg_pools
        
    def __call__(self, inputs, *args, **kwds):
        return self.forward(inputs, *args, **kwds)
    
    def backward(self, dvalues):
        # dvalues: N, C, H, W
        N, C_out, H_out, W_out = dvalues.shape
        davg_pool = dvalues / (self.pool_h * self.pool_w)
        davg_pool = davg_pool[:,:,None, :, :]
        davg_pool = np.repeat(davg_pool, repeats=self.pool_h*self.pool_w, axis=2)
        
        # davg_pool: [batch_size, C_in * pool_h * pool_w, H_out * W_out]
        davg_pool = davg_pool.reshape(N, C_out * self.pool_h * self.pool_w, H_out*W_out)
        davg_pool = col2im(davg_pool, 
                         (N, self.C_in, self.H_in, self.W_in), 
                         self.pool_h,
                         self.pool_w,
                         self.padding,
                         self.stride)
        return davg_pool
        
if __name__=='__main__':
    x = np.arange(96).reshape(2,1,6,8)
    pool = AvgPooling(2,2,2,0)
    out = pool(x)
    back = pool.backward(out)
    print(out)
    pass