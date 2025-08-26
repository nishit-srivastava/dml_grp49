import numpy as np
from . import masking

class Client:
    def __init__(self, name, data, labels=None):
        self.name = name
        self.data = data
        self.labels = labels  # Only one client may have labels
        self.weights = np.random.randn(data.shape[1])
    
    def compute_logits(self):
        return self.data @ self.weights
    
    def masked_logits(self):
        logits = self.compute_logits()
        return masking.add_mask(logits)
    
    def update(self, gradient, lr=0.01):
        self.weights -= lr * gradient
