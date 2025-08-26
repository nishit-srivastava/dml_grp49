import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 100

# Generate random features for Client A (5 features)
clientA = pd.DataFrame(
    np.random.rand(n_samples, 5),
    columns=[f"feat{i}" for i in range(1, 6)]
)

# Generate random features for Client B (5 features)
clientB = pd.DataFrame(
    np.random.rand(n_samples, 5),
    columns=[f"feat{i}" for i in range(6, 11)]
)

# Generate binary labels (based on some rule + noise for realism)
labels = pd.DataFrame({
    "label": ((clientA["feat1"] + clientB["feat6"] > 1) * 1).values
})

# Save to CSV files
clientA.to_csv("../data/clientA.csv", index=False)
clientB.to_csv("../data/clientB.csv", index=False)
labels.to_csv("../data/labels.csv", index=False)

print("✅ Synthetic data generated and saved as clientA.csv, clientB.csv, labels.csv")
