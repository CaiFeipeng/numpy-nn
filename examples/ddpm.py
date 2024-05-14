try:
    import cupy as np
except:
    import numpy as np
from src.dataset.cifar import Cifar10
import src.nn as nn
import time
from models.diffusion import Diffusion
from models.unet import Unet

def train():
    epochs = 60
    unet = Unet(image_channels=3)
    model = Diffusion(model=unet, 
                      timesteps=300, 
                      beta_start=0.0001, 
                      beta_end=0.02, 
                      loss_func=nn.MSELoss())
    
    optmizer = nn.SGDOptimizer(model.layers, learning_rate=0.0001, momentum=0.9, decay=0.1)

    dataloader = Cifar10(data_root='./data/cifar-10-batches-py', batch_size=10)
    
    for ep in range(epochs):
        start_time = time.time()
        for iter, (batch, label) in enumerate(dataloader):
            batch, label = np.asarray(batch), np.asarray(label)
            
            preds, targets = model(batch)
            dloss, loss = model.loss_func(preds, targets)
            
            model.backward(dloss) # loss.backward()
            optmizer.update()
            if iter % 10 == 0:
                end_time = time.time()
                print('loss: {},  {:.2f}s/iter'.format(loss, (end_time-start_time)/(iter+1)))
            

if __name__ == '__main__':
    train()