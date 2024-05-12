try:
    import cupy as np
except:
    import numpy as np
    
from models.transformer import Transformer
from src.dataset.multi30k import Multi30k
from src.nn.transformers import Encoder, Decoder
from src.nn import optimizers, losses
import time

PAD_TOKEN = '<pad>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'

PAD_INDEX = 0
SOS_INDEX = 1
EOS_INDEX = 2
UNK_INDEX = 3

tokens  = (PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN)
indexes = (PAD_INDEX, SOS_INDEX, EOS_INDEX, UNK_INDEX)

SRC_VOCAB_SIZE = None
TGT_VOCAB_SIZE = None
HID_DIM = 256  #512 in original paper
ENC_LAYERS = 3 #6 in original paper
DEC_LAYERS = 3 #6 in original paper
ENC_HEADS = 8
DEC_HEADS = 8
FFN_DIM = 512  #2048 in original paper
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
MAX_LEN = 5000

EPOCHES = 60
def train():

    dataloader = Multi30k(data_root='./data/multi30k',
                          tokens=tokens, indexes=indexes, batch_size=50, min_freq=2)
    SRC_VOCAB_SIZE = len(dataloader.get_vocabs()[0])
    TGT_VOCAB_SIZE = len(dataloader.get_vocabs()[1])

    encoder = Encoder(SRC_VOCAB_SIZE, ENC_HEADS, ENC_LAYERS, HID_DIM, FFN_DIM, ENC_DROPOUT, MAX_LEN)
    decoder = Decoder(TGT_VOCAB_SIZE, DEC_HEADS, DEC_LAYERS, HID_DIM, FFN_DIM, DEC_DROPOUT, MAX_LEN)

    model = Transformer(encoder, decoder, PAD_INDEX)

    optmizer = optimizers.SGDOptimizer(model.layers, learning_rate=0.0001, momentum=0.9, decay=0.1)
    loss_func = losses.CrossEntropy(num_classes=TGT_VOCAB_SIZE)

    for ep in range(EPOCHES):
        print("Epoch: {}".format(ep))
        start_time = time.time()
        for iter, (src, tgt) in enumerate(dataloader):
            src, tgt = np.asarray(src), np.asarray(tgt)
            output = model(src, tgt)
            dloss, loss = loss_func(output, tgt)

            model.backward(dloss)
            optmizer.update()
            if iter % 10 == 0:
                end_time = time.time()
                print('loss: {},  {:.2f}s/iter'.format(loss, (end_time-start_time)/(iter+1)))

if __name__=='__main__':
    train()