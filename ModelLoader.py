import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, n_filters=32, kernel_sizes=[9, 19, 39], dropout_rate=0.4):
        super(InceptionBlock, self).__init__()

        # Parallel convolutional layers with different kernel sizes
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, n_filters, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(n_filters),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ) for kernel_size in kernel_sizes
        ])

        # MaxPool branch
        self.maxpool = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_channels, n_filters, 1),
            nn.BatchNorm1d(n_filters),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Batch normalization and activation
        self.bn = nn.BatchNorm1d(n_filters * (len(kernel_sizes) + 1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Process through parallel convolution layers
        conv_outputs = [conv(x) for conv in self.conv_layers]

        # Process through maxpool branch
        max_output = self.maxpool(x)

        # Concatenate all branches
        concat = torch.cat(conv_outputs + [max_output], dim=1)

        # Apply batch normalization, activation, and dropout
        output = self.dropout(self.relu(self.bn(concat)))
        return output


class InceptionTime(nn.Module):
    def __init__(self, input_size, num_classes, num_blocks, n_filters=32, dropout_rate=0.4):
        super(InceptionTime, self).__init__()

        # Initial convolution layer with regularization
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, n_filters, 1),
            nn.BatchNorm1d(n_filters),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Calculate channels after each inception block (4 parallel paths * n_filters)
        inception_channels = n_filters * 4

        self.inception_blocks = nn.ModuleList([
            InceptionBlock(n_filters if i == 0 else inception_channels, n_filters, dropout_rate=dropout_rate)
            for i in range(num_blocks)
        ])

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers with dropout
        self.classifier = nn.Sequential(
            nn.Linear(inception_channels, inception_channels // 2),
            nn.BatchNorm1d(inception_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(inception_channels // 2, num_classes)
        )

        # Shortcut connections with regularization
        self.shortcuts = nn.ModuleList()
        shortcut_in_channels = [n_filters] + [inception_channels] * ((num_blocks // 3) - 1)
        for in_channels in shortcut_in_channels:
            self.shortcuts.append(nn.Sequential(
                nn.Conv1d(in_channels, inception_channels, 1),
                nn.BatchNorm1d(inception_channels),
                nn.Dropout(dropout_rate)
            ))

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # Reshape input to (batch_size, channels, length)
        x = x.view(x.size(0), 1, -1)

        # Initial convolution
        x = self.conv1(x)

        # Process through Inception blocks with residual connections
        shortcut_counter = 0
        for i, inception_block in enumerate(self.inception_blocks):
            if i % 3 == 0:
                shortcut = x

            x = inception_block(x)

            # Add residual connection every third block
            if i % 3 == 2:
                shortcut = self.shortcuts[shortcut_counter](shortcut)
                x = x + shortcut
                shortcut_counter += 1

        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)

        # Classification layers
        x = self.classifier(x)
        return x


class CustomDataset(Dataset):
    def __init__(self, csv_path):
        """
        Initialize the dataset from a CSV file.

        Args:
            csv_path (str): Path to the CSV file containing the data
        """
        # Read the CSV file
        self.data = pd.read_csv(csv_path)

        # Assuming the last column is the label and all other columns are features
        self.features = self.data.iloc[:, :-1].values
        self.labels = self.data.iloc[:, -1].values

        # Add feature normalization
        #self.features = (self.features - self.features.mean(axis=0)) / (self.features.std(axis=0) + 1e-8)

    def __len__(self):
        """Return the total number of samples in the dataset"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset

        Args:
            idx (int): Index of the sample to fetch

        Returns:
            tuple: (features, label) where features is a tensor and label is an integer
        """
        # Convert features and label to tensors
        features = torch.FloatTensor(self.features[idx])
        label = torch.LongTensor([self.labels[idx]])[0]

        return features, label
