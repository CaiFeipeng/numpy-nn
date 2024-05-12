try:
    import cupy as np
except:
    import numpy as np
from src.nn.linear import Linear
from src.nn.dropout import Dropout
from src.nn.activations import Sigmoid, SoftMax
from .base import Layer

class MultiHeadAttention(Layer):
    """Multi-Head Attention"""
    def __init__(self, input_d=512, heads_num=8, dropout=0.):
        self.input_d = input_d
        self.heads_num = heads_num
        self.head_dim = input_d // heads_num
        
        self.scale = np.sqrt(self.head_dim)
        
        self.Q_linear = Linear(input_d=input_d, output_d=input_d, use_bias=False)
        self.K_linear = Linear(input_d=input_d, output_d=input_d, use_bias=False)
        self.V_linear = Linear(input_d=input_d, output_d=input_d, use_bias=False)
        self.O_linear = Linear(input_d=input_d, output_d=input_d, use_bias=False)
        
        self.softmax = SoftMax()
        self.dropout = Dropout(dropout)
        
    def split_heads_forward(self, x):
        # x: [batch_size, seq_len, embed_dim]
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1, self.heads_num, self.head_dim).transpose(0, 2, 1, 3) # [batch_size, heads_num, seq_len, head_dim]
    def split_heads_backward(self, x):
        # x: [batch_size, heads_num, seq_len, head_dim]
        batch_size = x.shape[0]
        return x.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.input_d)
    
    def merge_heads_forward(self, x):
        # x: [batch_size, heads_num, seq_len, head_dim]
        batch_size = x.shape[0]
        return x.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.input_d)
    def merge_heads_backward(self, x):
        # x: [batch_size, seq_len, embed_dim]
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1, self.heads_num, self.head_dim).transpose(0, 2, 1, 3)
        
    def forward(self, query, key=None, value=None, mask=None, training=True):
        if key is None: key = query
        if value is None: value = key
        # query, key, value: [batch_size, seq_len, emb_dim]
        batch_size = key.shape[0]
        
        self.query_len, self.key_len, self.value_len = query.shape[1], key.shape[1], value.shape[1]
        # print(self.query_len, self.key_len, self.value_len)

        # self.QKV: [batch_size, seq_len, embed_dim]
        self.Q = self.Q_linear(query)
        self.K = self.K_linear(key)
        self.V = self.V_linear(value)
        
        # self.QKV: [batch_size, heads_num, seq_len, head_dim]
        self.Q = self.split_heads_forward(self.Q)
        self.K = self.split_heads_forward(self.K)
        self.V = self.split_heads_forward(self.V)
        
        # scores: [batch_size, heads_num, seq_len, seq_len]
        scores = np.matmul(self.Q, self.K.transpose(0, 1, 3, 2)) / self.scale
        
        if mask is not None:
            self.mask = np.asarray(mask)
            
            # self.mask: [N, 1, 1, seq_len]
            self.mask = self.mask[:, None, ...]
            scores = np.where(self.mask == 0, float('-inf'), scores)
            
        # attention: [batch_size, heads_num, query_seq_len, key_seq_len]
        attention = self.softmax(scores)
        
        self.dropout_attention = self.dropout(attention, training=training)
        
        # outputs: [batch_size, heads_num, seq_len, head_dim]
        split_outputs = np.matmul(self.dropout_attention, self.V)
        # transpose heads
        merge_outputs = self.merge_heads_forward(split_outputs)
        
        outputs = self.O_linear(merge_outputs)
        return outputs, attention
    
    def __call__(self, query, key=None, value=None, *args, **kwds):
        return self.forward(query, key, value, *args, **kwds)
    
    def backward(self, dvalues):
        # dvalues: [batch_size, query_seq_len, embed_dim]
        dvalues = self.O_linear.backward(dvalues)
        
        # dvalues: [batch_size, heads_num, seq_len, head_dim]
        dvalues = self.merge_heads_backward(dvalues)
        
        # dV: [batch_size, heads_num, value_seq_len, head_dim]
        dV = np.matmul(self.dropout_attention.transpose(0,1,3,2), dvalues)
        
        # dAtten: [batch_size, heads_num, seq_len, seq_len] 
        # self.V: [batch_size, heads_num, seq_len, head_dim]
        dAtten = np.matmul(dvalues, self.V.transpose(0, 1, 3, 2))
        
        dDropout = self.dropout.backward(dAtten)
        dSoftmax = self.softmax.backward(dDropout)
        
        if self.mask is not None:
            dSoftmax = np.where(self.mask == 0, 0, dSoftmax)
            
        dSoftmax = dSoftmax / self.scale # [batch_size, heads_num, seq_len, seq_len]
        
        # dQ: [batch_size, heads_num, seq_len, head_dim]
        dQ = np.matmul(dSoftmax, self.K)
        # dK: [batch_size, heads_num, head_dim, seq_len]
        dK = np.matmul(self.Q.transpose(0, 1, 3, 2), dSoftmax)
        dK = dK.transpose(0, 1, 3, 2) # [batch_size, heads_num, seq_len, head_dim]
        
        dV = self.split_heads_backward(dV)
        dQ = self.split_heads_backward(dQ)
        dK = self.split_heads_backward(dK)
        
        dV = self.V_linear.backward(dV)
        dQ = self.Q_linear.backward(dQ)
        dK = self.K_linear.backward(dK)
        
        return dQ, dK, dV
    
    @property
    def params(self):
        params = self.Q_linear.params + self.K_linear.params + self.V_linear.params \
                + self.O_linear.params
        return params
    @property
    def grads(self):
        grads = self.Q_linear.grads + self.K_linear.grads + self.V_linear.grads \
                + self.O_linear.grads
        return grads