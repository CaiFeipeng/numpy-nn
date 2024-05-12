from src.dataset.cifar import Cifar10
from src.dataset.toy_dataset import MultiClassDataset
import src.nn as nn
import time

def train():
    epochs = 60
    iters = 10000
    
    model = nn.Sequential([
        nn.Conv2D(3, 8, (3,3), stride=1, padding=1),
        nn.BatchNorm2D(8),
        nn.ReLU(),
        nn.Conv2D(8, 8, (3,3), stride=1, padding=1),
        nn.BatchNorm2D(8),
        nn.ReLU(),
        nn.AvgPooling(2, 2, 2),
        nn.Conv2D(8, 4, (3,3), stride=1, padding=1),
        nn.BatchNorm2D(4),
        nn.ReLU(),
        nn.AvgPooling(2, 2, 2),
        nn.Flatten(),
        nn.Linear(256, 10),
        nn.SoftMax()
    ])
    
    optmizer = nn.SGDOptimizer(model.layers, learning_rate=0.01, decay=0.1)
    loss_func = nn.CrossEntropy(num_classes=10)
    dataloader = Cifar10(data_root='./data/cifar-10-batches-py', batch_size=8)
    
    for ep in range(epochs):
        start_time = time.time()
        for iter, (batch, label) in enumerate(dataloader):
            # batch = batch.reshape(10,3,8,8)
            output = model(batch)
            dloss, loss = loss_func(output, label)
            # optmizer.zero_grad()
            model.backward(dloss) # loss.backward()
            optmizer.update()
            if iter % 10 == 0:
                end_time = time.time()
                print('loss: {},  {:.2f}s/iter'.format(loss, (end_time-start_time)/(iter+1)))
            

if __name__ == '__main__':
    train()