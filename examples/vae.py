from src.dataset.cifar import Cifar10
from src.dataset.toy_dataset import MultiClassDataset
import src.nn as nn
import time
from models.vae import VAE
try:
    import cupy as np
except:
    import numpy as np

def train():
    epochs = 60
    
    model = VAE(image_channels=3)
    
    optmizer = nn.SGDOptimizer(model.layers, learning_rate=0.0001, decay=0.1)

    dataloader = Cifar10(data_root='./data/cifar-10-batches-py', batch_size=2)
    
    for ep in range(epochs):
        start_time = time.time()
        for iter, (batch, label) in enumerate(dataloader):
            batch, label = np.asarray(batch), np.asarray(label)
            encoded, decoded = model(batch)
            drecons, dmu, dlog_var, loss = model.loss_func(decoded, batch)
            
            model.backward(drecons, dmu, dlog_var) # loss.backward()
            optmizer.update()
            if iter % 10 == 0:
                end_time = time.time()
                print('loss: {},  {:.2f}s/iter'.format(loss, (end_time-start_time)/(iter+1)))
            

if __name__ == '__main__':
    train()