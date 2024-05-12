import src.nn as nn
from src.dataset.toy_dataset import MultiClassDataset

def train():
    epochs = 60
    iters = 10000
    
    model = nn.Sequential([
        nn.Linear(20,128),
        nn.ReLU(),
        nn.Dropout(),
        nn.LayerNorm(128),
        nn.Linear(128,128),
        nn.ReLU(),
        nn.Dropout(),
        nn.LayerNorm(128),
        nn.Linear(128,128),
        nn.ReLU(),
        nn.Dropout(),
        nn.LayerNorm(128),
        nn.Linear(128,2),
        nn.SoftMax(),
    ])
    
    optmizer = nn.SGDOptimizer(model.layers, learning_rate=0.001, decay=0.1)
    # loss_func = losses.MSELoss()
    loss_func = nn.CrossEntropy(num_classes=2)
    
    # dataloader = MultivarLinearDataset(input_d=20, batchsize=20, num_samples=1000)
    dataloader = MultiClassDataset(num_classes=2, input_d=20, batchsize=20, num_samples=1000)
    
    for ep in range(epochs):
        for iter, (batch, label) in enumerate(dataloader):
            output = model(batch)
            dloss, loss = loss_func(output, label)
            # optmizer.zero_grad()
            model.backward(dloss) # loss.backward()
            optmizer.update()
            if iter % 1000 == 0:
                print('loss: ', loss)
            

if __name__ == '__main__':
    train()