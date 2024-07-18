import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import pandas as pd
number_of_features=4 # change the number_of_features according to the data-set
class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.data.iloc[idx, :number_of_features].values, dtype=torch.float32)
        class_name = torch.tensor(self.data.iloc[idx, number_of_features], dtype=torch.long)

        if self.transform:
            features = self.transform(features)

        return features, class_name
