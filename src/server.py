import numpy as np
from . import masking

class Server:
    def __init__(self, clients):
        self.clients = clients
    
    def aggregate(self):
        # Collect masked logits from each client
        masked_logits = [c.masked_logits() for c in self.clients]
        combined = np.sum(masked_logits, axis=0)
        # Unmask
        return masking.remove_mask(combined)
    
    def compute_loss_and_gradients(self, y_true, logits):
        preds = 1 / (1 + np.exp(-logits))
        loss = -np.mean(y_true * np.log(preds + 1e-9) + (1-y_true)*np.log(1-preds+1e-9))
        grad_common = preds - y_true
        grads = []
        for c in self.clients:
            grads.append(c.data.T @ grad_common / len(y_true))
        return loss, grads
