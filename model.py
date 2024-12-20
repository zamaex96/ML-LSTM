import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import pandas as pd

# Define the MLP model
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = self.fc(lstm_out)
        x = self.softmax(x)
        return x

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Check input dimensions
        if x.dim() == 2:  # If input is 2D (batch_size, input_size)
            batch_size = x.size(0)
            x = x.unsqueeze(1)  # Add seq_length dimension (1 time step)

        # Forward propagate RNN
        out, _ = self.rnn(x)  # out shape: (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        if out.dim() == 3:
            out = self.fc(out[:, -1, :])  # Use the last time step
        else:
            out = self.fc(out)  # Directly use output if only 2D

        return out

# Define 1D CNN Model
class CNN1DModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNN1DModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)  # Convolution layer
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)  # Pooling layer
        self.fc1 = nn.Linear(32 * (input_size // 2 // 2), 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, num_classes)  # Output layer

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # Conv1 + ReLU + Pooling
        x = self.pool(torch.relu(self.conv2(x)))  # Conv2 + ReLU + Pooling
        x = x.view(x.size(0), -1)  # Flatten the output
        x = torch.relu(self.fc1(x))  # Fully connected layer with ReLU
        x = self.fc2(x)  # Output layer
        return x

