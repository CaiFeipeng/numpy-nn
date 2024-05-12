import numpy as np
import pickle
import os

def unpickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data

class Cifar10:
    def __init__(self, data_root, batch_size=1) -> None:
        self.data_root = data_root
        self.batch_size = batch_size
        self.samples = self.load_samples()
        self.num_samples = len(self.samples[0])
        self.train_batches = np.array_split(self.samples[0], np.arange(batch_size, self.num_samples, batch_size))
        self.train_labels = np.array_split(self.samples[1], np.arange(batch_size, self.num_samples, batch_size))
        
    def load_samples(self):
        self.meta_data = unpickle(os.path.join(self.data_root, 'batches.meta'))
        cifar_label_names = self.meta_data[b'label_names']
        cifar_label_names = np.array(cifar_label_names)
        
        cifar_train_data = None
        self.cifar_train_filenames = []
        cifar_train_labels = []
        
        for i in range(1, 6):
            cifar_train_data_dict = unpickle(os.path.join(self.data_root + "/data_batch_{}".format(i)))
            if i == 1:
                cifar_train_data = cifar_train_data_dict[b'data']
            else:
                cifar_train_data = np.vstack((cifar_train_data, cifar_train_data_dict[b'data']))
            self.cifar_train_filenames += cifar_train_data_dict[b'filenames']
            cifar_train_labels += cifar_train_data_dict[b'labels']

        cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))
        
        return cifar_train_data, cifar_train_labels
    
    def __getitem__(self, index):
        inputs, labels = [], []
        inputs = self.train_batches[index]
        labels = self.train_labels[index]
        inputs = np.array(inputs) / 255
        labels = np.array(labels)
        labels = np.expand_dims(labels, axis=1)
        return inputs, labels
    
if __name__=='__main__':
    dataset=Cifar10(data_root='./data/cifar10/cifar-10-batches-py', batch_size=2)
    for ep in range(1):
        for input, label in dataset:
            print(input, label)