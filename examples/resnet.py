try:
    import cupy as np
except:
    import numpy as np
from src.dataset.cifar import Cifar10
from src.dataset.toy_dataset import MultiClassDataset
import src.nn as nn
from models.resnet import ResNet18
import time

def train():
    epochs = 60
    
    model = ResNet18(image_channels=3, num_classes=10)
    
    optmizer = nn.SGDOptimizer(model.layers, learning_rate=0.0001, decay=0.1)
    loss_func = nn.CrossEntropy(num_classes=10)
    dataloader = Cifar10(data_root='./data/cifar-10-batches-py', batch_size=8)
    
    for ep in range(epochs):
        start_time = time.time()
        for iter, (batch, label) in enumerate(dataloader):
            batch, label = np.asarray(batch), np.asarray(label)
            output = model(batch)
            dloss, loss = loss_func(output, label)
            
            model.backward(dloss) # loss.backward()
            optmizer.update()
            if iter % 1 == 0:
                end_time = time.time()
                print('loss: {},  {:.2f}s/iter'.format(loss, (end_time-start_time)/(iter+1)))
            

if __name__ == '__main__':
    train()