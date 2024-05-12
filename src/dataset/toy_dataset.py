import numpy as np
from sklearn.datasets import make_classification

def linear_func(x, weights=None):
    if not weights:
        weights = np.array(list(range(1, len(x[0])+1)))
        
    return np.dot(x, weights)
class MultivarLinearDataset:
    def __init__(self, linear_func=linear_func, input_d=10, batchsize=2, num_samples=10000) -> None:
        self.linear_func = linear_func
        self.input_d = input_d
        self.num_samples = num_samples
        self.batchsize = batchsize
        self.samples = self.generate_samples()
    
    def __len__(self):
        return self.num_samples
        
    def generate_samples(self):
        x = np.random.randint(low=0,high=10, size=(self.num_samples, self.input_d)).astype(np.float32)
        y = linear_func(x)
        epsilon = np.random.normal(0, 0.1, self.num_samples)
        
        y += epsilon
        return x, y
    
    def __getitem__(self, index):
        inputs, labels = [], []
        for _ in range(self.batchsize):
            index = np.random.randint(0, self.num_samples-1)
            inputs.append(self.samples[0][index])
            labels.append(self.samples[1][index])
        inputs = np.array(inputs)
        labels = np.array(labels)
        labels = np.expand_dims(labels, axis=1)
        return inputs, labels
    

class MultiClassDataset:
    def __init__(self, num_classes, input_d, num_samples, batchsize = 2) -> None:
        self.num_classes = num_classes
        self.input_d = input_d
        self.num_samples = num_samples
        self.batchsize = batchsize
        self.samples = self.generate_samples()
        
    def generate_samples(self):
        x, y = make_classification(n_samples=self.num_samples, n_features=self.input_d, n_classes=self.num_classes)
        return x, y
    
    def __getitem__(self, index):
        inputs, labels = [], []
        for _ in range(self.batchsize):
            index = np.random.randint(0, self.num_samples)
            inputs.append(self.samples[0][index])
            labels.append(self.samples[1][index])
        inputs = np.array(inputs)
        labels = np.array(labels)
        labels = np.expand_dims(labels, axis=1)
        return inputs, labels
        
if __name__=='__main__':
    dataset=MultivarLinearDataset(input_d=2, num_samples=10)
    for ep in range(1):
        for input, label in dataset:
            print(input, label)