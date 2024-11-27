import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os

class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Custom Dataset for loading CSV data.

        Args:
            csv_file (str): Path to the CSV file.
            transform (callable, optional): Optional transform to apply to the data.
        """
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file '{csv_file}' not found.")
        
        # Load the data
        self.data = pd.read_csv(csv_file)
        if self.data.empty:
            raise ValueError("CSV file is empty or improperly formatted.")

        self.num_features = self.data.shape[1] - 1  # All but the last column are features
        self.transform = transform

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve a sample and its corresponding label.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Features and class label.
        """
        # Extract features and label
        features = torch.tensor(self.data.iloc[idx, :self.num_features].values, dtype=torch.float32)
        label = torch.tensor(self.data.iloc[idx, self.num_features], dtype=torch.long)

        if self.transform:
            features = self.transform(features)

        return features, label
