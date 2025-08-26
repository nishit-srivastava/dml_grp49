import numpy as np
from src.utils import load_and_split
from src.client import Client
from src.server import Server

# Load data
X_train, X_test, y_train, y_test = load_and_split('data/toy.csv')

# Split into two views (non-overlapping features)
Xa, Xb = X_train[:, :3], X_train[:, 3:]
Xta, Xtb = X_test[:, :3], X_test[:, 3:]

clientA = Client('A', Xa, labels=y_train)
clientB = Client('B', Xb)
server = Server([clientA, clientB])

# Train
for epoch in range(10):
    logits = server.aggregate()
    loss, grads = server.compute_loss_and_gradients(y_train, logits)
    clientA.update(grads[0])
    clientB.update(grads[1])
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
