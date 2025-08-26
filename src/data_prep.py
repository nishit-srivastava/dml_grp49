import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

def generate_dataset(path='data/toy.csv'):
    X, y = make_classification(n_samples=200, n_features=6, n_informative=4, random_state=42)
    df = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
    df['label'] = y
    df.to_csv(path, index=False)
    print(f"Dataset saved to {path}")

if __name__ == "__main__":
    generate_dataset()
