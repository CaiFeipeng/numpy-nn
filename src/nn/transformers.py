try:
    import cupy as np
except:
    import numpy as np
from src.nn.linear import Linear
from src.nn.activations import ReLU, SoftMax
from src.nn.dropout import Dropout
from src.nn.attention import MultiHeadAttention
from src.nn.norms import LayerNorm
from src.nn.positional_encoding import PositionalEncoding
from src.nn.embedding import Embedding
from .base import Layer

class FFN(Layer):
    """Feedforward Network"""
    def __init__(self, input_d=512, hidden_dim=2048, dropout=0.):

        self.fc_1 = Linear(input_d=input_d, output_d=hidden_dim)
        self.relu = ReLU()
        self.fc_2 = Linear(input_d=hidden_dim, output_d=input_d)
        self.dropout = Dropout(dropout)
        
    def forward(self, x, training=True):
        x = self.fc_1(x)
        x = self.relu(x)
        x = self.dropout(x, training=training)
        x = self.fc_2(x)
        return x
    
    def __call__(self, query, training=True, *args, **kwds):
        return self.forward(query, training, *args, **kwds)
    
    def backward(self, dvalues):
        dvalues = self.fc_2.backward(dvalues)
        dvalues = self.dropout.backward(dvalues)
        dvalues = self.relu.backward(dvalues)
        dvalues = self.fc_1.backward(dvalues)
        return dvalues
    @property
    def params(self):
        params = self.fc_1.params + self.fc_2.params
        return params
    @property
    def grads(self):
        grads = self.fc_1.grads + self.fc_2.grads
        return grads
    
class EncoderLayer(Layer):
    def __init__(self, input_d, ffn_d, heads_num=8, dropout=0.):
        self.layer_norm_1 = LayerNorm(input_d=input_d)
        self.layer_norm_2 = LayerNorm(input_d=input_d)
        self.self_attention  = MultiHeadAttention(input_d=input_d, heads_num=heads_num, dropout=dropout)
        self.ffn = FFN(input_d=input_d, hidden_dim=ffn_d, dropout=dropout)
        self.dropout_1 = Dropout(dropout)
        self.dropout_2 = Dropout(dropout)
        
    def forward(self, query, mask=None, training=True):
        _query, _ = self.self_attention(query, mask = mask, training=training)
        query = self.layer_norm_1(query + self.dropout_1(_query, training=training))
        
        _query = self.ffn(query=query, training=training)
        query = self.layer_norm_2(query + self.dropout_2(_query, training=training))
        return query
    
    def __call__(self, query, mask=None, training=True, *args, **kwds):
        return self.forward(query, mask, training, *args, **kwds)
    
    def backward(self, dvalues):
        dvalues = self.layer_norm_2.backward(dvalues)
        
        d_dropout = self.dropout_2.backward(dvalues)
        d_query = self.ffn.backward(d_dropout)
        
        dvalues = dvalues + d_query
        dvalues = self.layer_norm_1.backward(dvalues)
        
        d_dropout = self.dropout_1.backward(dvalues)
        dQ, dK, dV = self.self_attention.backward(d_dropout)
        
        return dvalues + dQ + dK + dV
    @property
    def params(self):
        params = self.layer_norm_1.params + self.layer_norm_2.params + self.self_attention.params \
                + self.ffn.params
        return params
    @property
    def grads(self):
        grads = self.layer_norm_1.grads + self.layer_norm_2.grads + self.self_attention.grads \
                + self.ffn.grads
        return grads
        
class DecoderLayer(Layer):
    def __init__(self, input_d, heads_num, ffn_d, dropout=0.):
        self.layer_norm_1 = LayerNorm(input_d=input_d)
        self.layer_norm_2 = LayerNorm(input_d=input_d)
        self.layer_norm_3 = LayerNorm(input_d=input_d)
        
        self.self_attention = MultiHeadAttention(input_d=input_d, heads_num=heads_num, dropout=dropout)
        self.cross_attention = MultiHeadAttention(input_d=input_d, heads_num=heads_num, dropout=dropout)
        
        self.ffn = FFN(input_d=input_d, hidden_dim=ffn_d, dropout=dropout)
        
        self.dropout_1 = Dropout(dropout)
        self.dropout_2 = Dropout(dropout)
        self.dropout_3 = Dropout(dropout)
        
    def forward(self, query, key, query_mask=None, key_mask=None, training=True):
        # self attention
        _query, _ = self.self_attention(query, query, query, mask=query_mask, training=training)
        query = self.layer_norm_1(query + self.dropout_1(_query, training=training))
        # cross attention
        _query, attention = self.cross_attention(query, key, key, mask=key_mask, training=training)
        query = self.layer_norm_2(query + self.dropout_2(_query, training=training))
        # feedforward
        _query = self.ffn(query, training=training)
        query = self.layer_norm_3(query + self.dropout_3(_query, training=training))
        
        return query, attention
    
    def __call__(self, query, key, query_mask=None, key_mask=None, training=True, *args, **kwds):
        return self.forward(query, key, query_mask, key_mask, training, *args, **kwds)
    
    def backward(self, dvalues):
        # feedforward backward
        dvalues = self.layer_norm_3.backward(dvalues)
        d_dropout_3 = self.dropout_3.backward(dvalues)
        d_ffn = self.ffn.backward(d_dropout_3)
        dvalues = dvalues + d_ffn
        # cross attention backward
        dvalues = self.layer_norm_2.backward(dvalues)
        d_dropout_2 = self.dropout_2.backward(dvalues)
        d_cross_attn_Q, d_cross_attn_K, d_cross_attn_V = self.cross_attention.backward(d_dropout_2)
        dvalues = dvalues + d_cross_attn_Q
        # self attention backward
        dvalues = self.layer_norm_1.backward(dvalues)
        d_dropout_1 = self.dropout_1.backward(dvalues)
        d_self_attn_Q, d_self_attn_K, d_self_attn_V = self.self_attention.backward(d_dropout_1)
        
        return d_self_attn_Q + d_self_attn_K + d_self_attn_V + dvalues, d_cross_attn_K + d_cross_attn_V
    
    @property
    def params(self):
        params = self.layer_norm_1.params + self.layer_norm_2.params + self.layer_norm_3.params \
                + self.self_attention.params + self.cross_attention.params + self.ffn.params
        return params
    @property
    def grads(self):
        grads = self.layer_norm_1.grads + self.layer_norm_2.grads + self.layer_norm_3.grads \
                + self.self_attention.grads + self.cross_attention.grads + self.ffn.grads
        return grads
        
class Encoder(Layer):
    """Transformer Encoder"""
    def __init__(self, src_vocab_size, heads_num, layer_num, d_model, ffn_d, dropout, max_len=5000) -> None:
        self.layer_num = layer_num
        
        self.tokenizer = Embedding(input_d=src_vocab_size, output_d=d_model)
        self.position_embedding = PositionalEncoding(max_len, output_d=d_model, dropout=dropout)
        
        self.layers = []
        for _ in range(layer_num):
            self.layers.append(EncoderLayer(input_d=d_model, ffn_d=ffn_d, heads_num=heads_num, dropout=dropout))
            
        self.dropout = Dropout(dropout)
        self.scale = np.sqrt(d_model)
        
    def forward(self, src, query_mask=None, training=True):
        src = self.tokenizer(src) * self.scale
        src = self.position_embedding(src)
        src = self.dropout(src, training=training)
        
        for layer in self.layers:
            src = layer(src, mask=query_mask, training=training)
            
        return src
    def __call__(self, src, query_mask=None, training=True, *args, **kwds):
        return self.forward(src, query_mask, training, *args, **kwds)
    
    def backward(self, dvalues):
        
        for layer in reversed(self.layers):
            dvalues = layer.backward(dvalues)
            
        dvalues = self.dropout.backward(dvalues)
        dvalues = self.position_embedding.backward(dvalues)
        dvalues = dvalues * self.scale
        dvalues = self.tokenizer.backward(dvalues)
        return dvalues
    
    @property
    def params(self):
        params = []
        for layer in reversed(self.layers):
            params += layer.params
        params += self.tokenizer.params
        return params
    
    @property
    def grads(self):
        grads = []
        for layer in reversed(self.layers):
            grads += layer.grads
        grads += self.tokenizer.grads
        return grads
    
class Decoder(Layer):
    def __init__(self, tgt_vocab_size, heads_num, layer_num, d_model, ffn_d, dropout, max_len=5000) -> None:
        self.tokenizer = Embedding(tgt_vocab_size, d_model)
        self.position_embedding = PositionalEncoding(max_len, d_model, dropout=dropout)
        
        self.layers = []
        for _ in range(layer_num):
            self.layers.append(DecoderLayer(input_d=d_model,heads_num=heads_num, ffn_d=ffn_d, dropout=dropout))
            
        self.fc_out = Linear(input_d=d_model, output_d=tgt_vocab_size)
        self.softmax = SoftMax()
        self.dropout = Dropout(dropout)
        self.scale = np.sqrt(d_model)
        
    def forward(self, tgt, src, tgt_mask=None, src_mask=None, training=True):
        tgt = self.tokenizer(tgt) * self.scale
        tgt = self.position_embedding(tgt)
        tgt = self.dropout(tgt, training=training)
        
        for layer in self.layers:
            tgt, attention = layer(tgt, src, query_mask=tgt_mask, key_mask=src_mask, training=training)
            
        outputs = self.fc_out(tgt)
        outputs = self.softmax(outputs)
        return outputs, attention
    def __call__(self, tgt, src, tgt_mask=None, src_mask=None, training=True, *args, **kwds):
        return self.forward(tgt, src, tgt_mask, src_mask, training, *args, **kwds)
    
    def backward(self, dvalues):
        dvalues = self.softmax.backward(dvalues)
        dvalues = self.fc_out.backward(dvalues)
        
        self.encoder_dvalues = 0
        for layer in reversed(self.layers):
            dvalues, encoder_dvalues = layer.backward(dvalues)
            self.encoder_dvalues += encoder_dvalues
        
        dvalues = self.dropout.backward(dvalues)
        dvalues = self.position_embedding.backward(dvalues)
        dvalues = dvalues * self.scale
        dvalues = self.tokenizer.backward(dvalues)
        return dvalues, self.encoder_dvalues
    
    @property
    def params(self):
        params = []
        params += self.fc_out.params
        
        for layer in reversed(self.layers):
            params += layer.params
        
        params += self.tokenizer.params
        return params
    
    @property
    def grads(self):
        grads = []
        grads += self.fc_out.grads
        
        for layer in reversed(self.layers):
            grads += layer.grads
            
        grads += self.tokenizer.grads
        return grads
    