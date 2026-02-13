import torch
from torch.utils.data import Dataset
import numpy as np

class DrugResponseDataset(Dataset):
    def __init__(self, n_samples=500):
        self.drug_features = np.random.rand(n_samples, 128)
        self.rna_features = np.random.rand(n_samples, 500)
        self.labels = np.random.rand(n_samples, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.drug_features[idx], dtype=torch.float32),
            torch.tensor(self.rna_features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )
