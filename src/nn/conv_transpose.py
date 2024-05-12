try:
    import cupy as np
except:
    import numpy as np
from typing import Any
from src.utils.im2col import im2col, col2im
from src.nn.base import Layer

class ConvTranspose2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, use_bias=True):
        self.C_in = in_channels
        self.C_out = out_channels
        self.kernel_height = kernel_size[0]
        self.kernel_width = kernel_size[1]
        self.use_bias = use_bias
        self.stride = stride
        self.padding = padding
        self.stride_1 = 1
        self.input_stride = self.stride - 1
        self.input_pad_h = self.kernel_height - self.padding - 1
        self.input_pad_w = self.kernel_width - self.padding - 1
        
        
        self.weights = 0.001 * np.random.randn(self.C_out, self.C_in, self.kernel_height, self.kernel_width)
        if self.use_bias:
            self.biases = np.zeros((1, self.C_out, 1, 1))
            
        
    def forward(self, inputs, training=True):
        N, C_in, self.H_in, self.W_in = inputs.shape
        self.H_out = (self.H_in - 1) * self.stride + self.kernel_height - 2 * self.padding
        self.W_out = (self.W_in - 1) * self.stride + self.kernel_width - 2 * self.padding
        
        self.H_in_padded = self.H_in + self.input_stride*(self.H_in-1) + 2*self.input_pad_h
        self.W_in_padded = self.W_in + self.input_stride*(self.W_in-1) + 2*self.input_pad_w
        # self.H_in_padded, self.W_in_padded = self.H_out + 2*self.input_pad_h, self.W_out + 2*self.input_pad_w
        inputs_padded = np.zeros((N, C_in, self.H_in_padded, self.W_in_padded))
        
        inputs_padded[:,:,
                      self.input_pad_h:self.H_out+self.input_pad_h:self.input_stride+1,
                      self.input_pad_w:self.W_out+self.input_pad_w:self.input_stride+1] = inputs
        
        # self.im2col = [batch_size, C_in*kernel_height*kernel_width, H_out*W_out]
        self.im2col = im2col(inputs_padded, self.kernel_height, self.kernel_width, padding=0, stride=self.stride_1)
        outputs = np.matmul(self.weights.reshape(self.C_out, -1), self.im2col)
        if self.use_bias:
            outputs = outputs.reshape(N, self.C_out, self.H_out, self.W_out) + self.biases
        else:
            outputs = outputs.reshape(N, self.C_out, self.H_out, self.W_out)
        return outputs
    
    def __call__(self, inputs, training=True, *args, **kwds):
        return self.forward(inputs, training, *args, **kwds)
        
    def backward(self, dvalues):
        # dvalues [batch_size, C_out, H_out, W_out]
        N = dvalues.shape[0]
        
        if self.use_bias:
            self.dbiases = np.sum(dvalues, axis=(0, 2, 3), keepdims=True)
            
        dvalues_reshaped = dvalues.reshape(N, self.C_out, -1)
        dweights_reshaped = np.matmul(dvalues_reshaped, self.im2col.transpose(0, 2, 1))
        dweights_reshaped = np.sum(dweights_reshaped, axis=0)
        self.dweights = dweights_reshaped.reshape(self.C_out, self.C_in, self.kernel_height, self.kernel_width)
        
        dinputs_im2col = np.matmul(self.weights.reshape(self.C_out,-1).T, dvalues_reshaped)
        dinputs_padded = col2im(dinputs_im2col,
                                (N, self.C_in, self.H_in_padded, self.W_in_padded),
                                self.kernel_height,
                                self.kernel_width,
                                padding=0,
                                stride=self.stride_1)
        
        dinputs = dinputs_padded[:,:,
                      self.input_pad_h:self.H_out+self.input_pad_h:self.input_stride+1,
                      self.input_pad_w:self.W_out+self.input_pad_w:self.input_stride+1]
        return dinputs
    
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
        
if __name__=='__main__':
    inputs = np.arange(12).reshape(1,1,3,4)
    convt = ConvTranspose2D(1,2,(3,3),stride=2,padding=1)
    out = convt(inputs)
    pass