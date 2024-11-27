import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
import numpy as np  # Add this line to import numpy
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from model import *
from data_loader import *

model_name = "LSTM"
ext = "TSNormOnly"

# Function to compute the confusion matrix
def plot_confusion_matrix(true_labels, predicted_labels, class_names):
    cm = confusion_matrix(true_labels, predicted_labels)
    # Normalize confusion matrix (to display percentages)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    accuracy = accuracy_score(true_labels, predicted_labels)
    plt.figure(figsize=(10, 8))  # Increase figure size
    ax = sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='coolwarm',  #cmap='viridis/coolwarm/inferno/magma/cividis/YlGnBu'
                     xticklabels=class_names, yticklabels=class_names,
                     linewidths=0.5, linecolor='black', cbar_kws={'label': 'Percentage'})

    # Increase font sizes
    plt.title(f'Confusion Matrix (Accuracy: {accuracy * 100:.2f}%)', fontsize=16)
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('True', fontsize=14)
    plt.xticks(fontsize=12, rotation=45)  # Rotate x-axis labels if they overlap
    plt.yticks(fontsize=12)

    # Add color bar label and grid lines
    cbar = ax.collections[0].colorbar
    cbar.set_label('Percentage', fontsize=12)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=12)
    ax.set_yticklabels(class_names, rotation=0, fontsize=12)

    # Show plot
    plt.tight_layout()
    output_folder = r"C:\Users\ML\Plots"
    os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist
    png_file_path = os.path.join(output_folder, f"{model_name}_{ext}.png")
    plt.savefig(png_file_path, format='png', dpi=600)  # Save as PNG
    plt.show()

# Model parameters
input_size = 8
hidden_size = 4
output_size = 4

# Paths to training and testing data
# model_path = r"C:\Users\ML\Models\LSTM.pth"
# model_path = r"C:\Users\ML\Models\Only.pth"
model_path = r"C:\Users\ML\Models\LSTM.pth"

#model_inference = RNNModel(input_size, hidden_size, num_layers, output_size)
#model_inference = SpatialAttentionModel(input_size, hidden_size, output_size, num_layers)
#model_inference = TransformerModel(input_size, hidden_size, output_size, num_layers=num_layers, nhead=n_head)
model_inference = LSTMModel(self, input_size, hidden_size, output_size)
model_inference.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_inference.to(device)

test_csv_path = r"C:\Users\Datatset\test.csv"
test_dataset = CustomDataset(test_csv_path)
test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model_inference.eval()
true_labels = []
predicted_labels = []

with torch.no_grad():
    for inputs, labels in test_data_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model_inference(inputs)
        print(f"Model output shape: {outputs.shape}")  # Print the shape of outputs: (batch_size, numberOfOutputs)
        _, predicted = torch.max(outputs.data, 1)

        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

class_names = ["C1", "C2", "C3", "C4"]

plot_confusion_matrix(true_labels, predicted_labels, class_names)
