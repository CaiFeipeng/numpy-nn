import numpy as np
from src.nn.transformers import Encoder, Decoder

class Transformer:
    def __init__(self, encoder, decoder, pad_idx) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.layers = [self.decoder, self.encoder]

    def get_pad_mask(self, x):
        #x: (batch_size, seq_len)
        return (x != self.pad_idx).astype(int)[:, np.newaxis, :]
    
    @classmethod
    def get_causal_mask(cls, x):
        #x: (batch_size, seq_len)
        seq_len = x.shape[1]
        causal_mask = np.triu(np.ones((seq_len, seq_len)), k = 1).astype(int)
        causal_mask = np.logical_not(causal_mask)
        return causal_mask
    
    def forward(self, src, tgt, training=True):
        src_mask = self.get_pad_mask(src)
        tgt_mask = self.get_pad_mask(tgt) & self.get_causal_mask(tgt)

        enc_src = self.encoder.forward(src, src_mask, training)

        out, attention = self.decoder.forward(tgt, enc_src, tgt_mask, src_mask, training)
        # output: (batch_size, target_seq_len, vocab_size)
        # attn: (batch_size, heads_num, target_seq_len, source_seq_len)
        return out
    
    def __call__(self, src, tgt, training=True, *args, **kwds):
        return self.forward(src, tgt, training, *args, **kwds)
    
    def backward(self, dvalues):
        dvalues = self.decoder.backward(dvalues)
        dvalues = self.encoder.backward(self.decoder.encoder_dvalues)
        return dvalues
    

if __name__=='__main__':
    x = np.arange(12).reshape(2,6)
    m = Transformer.get_causal_mask(x)
    pass