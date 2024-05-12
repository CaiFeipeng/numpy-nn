try:
    import cupy as np
except:
    import numpy as np

def softmax(inputs):
    # rowMax = np.max(inputs, axis=-1, keepdims=True)
    # inputs -= rowMax
    exp = np.exp(inputs)
    return exp / (exp.sum(axis=-1, keepdims=True)+1e-8)

class L1_loss:
    
    def forward(self, preds, labels):
        l1 = np.mean((preds - labels))
        return l1
    
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)
    
class MSELoss:
    def __call__(self, preds, labels):
        dloss = (preds - labels)/(len(labels))
        loss = np.mean((preds - labels)**2) / 2
        return dloss, loss
    
class CrossEntropy:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.eps = 1e-20
        
    def __call__(self, preds, labels):
        log_pred = np.log(preds + self.eps)
        # log_pred = np.clip(log_pred, a_min=-1e5, a_max=1e5)
        # if np.isnan(log_pred).any():
        #     print(log_pred)
        
        def mask(label, batch):
            log_mask = np.zeros((label.size, self.num_classes))
            log_mask[np.arange(label.size), label.reshape(1, -1)] = 1
            return log_mask.reshape(preds.shape)
        
        log_mask = mask(labels, preds.shape[0])
        loss = -1 * log_mask * log_pred
        
        dloss = -1 * log_mask / (preds + self.eps)
        
        
        return dloss, np.mean(loss)