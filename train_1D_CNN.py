import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from data_loader import CustomDataset  # Ensure CustomDataset is imported
from model import *

model_name = "1D-CNN"
ext="TS"

# Model parameters
input_size = 8  # Number of input features (length of each sample)
output_size = 4  # Number of output classes

# Instantiate the 1D CNN model
model = CNN1DModel(input_size, output_size)

# Set device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Paths to training and testing data
train_csv_path = r"C:\train.csv"
test_csv_path = r"C:\test.csv"

# Create datasets and dataloaders
train_dataset = CustomDataset(train_csv_path)
test_dataset = CustomDataset(test_csv_path)

train_data_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=12, shuffle=False)

# Training parameters
epochs = 1000
train_loss_values = []
test_loss_values = []
train_accuracy_values = []
test_accuracy_values = []

# Training loop
for epoch in range(epochs):
    model.train()
    epoch_train_loss = 0.0
    correct_train = 0
    total_train = 0
    for inputs, labels in train_data_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Reshape inputs for 1D CNN (batch_size, channels, input_length)
        inputs = inputs.unsqueeze(1)  # Add channel dimension

        optimizer.zero_grad()
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update parameters

        epoch_train_loss += loss.item()

        # Get predictions
        _, predicted_train = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()

    # Average loss and accuracy for the epoch
    epoch_train_loss /= len(train_data_loader)
    train_loss_values.append(epoch_train_loss)
    train_accuracy = 100 * correct_train / total_train
    train_accuracy_values.append(train_accuracy)

    # Testing phase
    model.eval()
    epoch_test_loss = 0.0
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for inputs, labels in test_data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Reshape inputs for 1D CNN
            inputs = inputs.unsqueeze(1)

            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            epoch_test_loss += loss.item()

            # Get predictions
            _, predicted_test = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted_test == labels).sum().item()

    # Average test loss and accuracy for the epoch
    epoch_test_loss /= len(test_data_loader)
    test_loss_values.append(epoch_test_loss)
    test_accuracy = 100 * correct_test / total_test
    test_accuracy_values.append(test_accuracy)

    if (epoch + 1) % 5 == 0:
        print(
            f'Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_train_loss:.4f}, Test Loss: {epoch_test_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')

# Specify output folder for saving the model, CSV file, and plots
output_folder1 = r"C:\Models" # Replace with desired path
os.makedirs(output_folder1, exist_ok=True)  # Create folder if it doesn't exist
output_folder2 = r"C:\CSV"  # Replace with desired path
os.makedirs(output_folder2, exist_ok=True)  # Create folder if it doesn't exist
output_folder3 = r"C:\Plots"  # Replace with desired path
os.makedirs(output_folder3, exist_ok=True)  # Create folder if it doesn't exist

# Save the model state
model_path = os.path.join(output_folder1, f"{model_name}_{ext}.pth")
torch.save(model.state_dict(), model_path)
print(f"Model saved at {model_path}")

# Save training information to a CSV file
train_info = {
    'train_loss': train_loss_values,
    'train_accuracy': train_accuracy_values,
    'test_loss': test_loss_values,
    'test_accuracy': test_accuracy_values
}
train_info_df = pd.DataFrame(train_info)
csv_path = os.path.join(output_folder2, f"{model_name}_{ext}.csv")
train_info_df.to_csv(csv_path, index=False)
print(f"Training data saved at {csv_path}")

# Plot the loss and accuracy over epochs
plt.figure(figsize=(12, 4))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_loss_values, label='Training Loss')
plt.plot(range(1, epochs + 1), test_loss_values, label='Testing Loss')
plt.title('Training and Testing Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), train_accuracy_values, label='Training Accuracy')
plt.plot(range(1, epochs + 1), test_accuracy_values, label='Testing Accuracy')
plt.title('Training and Testing Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()

# Save plots as PNG and PDF
png_file_path = os.path.join(output_folder3, f"{model_name}_{ext}.png")
pdf_file_path = os.path.join(output_folder3, f"{model_name}_{ext}.pdf")
plt.savefig(png_file_path, format='png', dpi=600)
plt.savefig(pdf_file_path, format='pdf', dpi=600)
print(f"Plots saved at {png_file_path} and {pdf_file_path}")

plt.show()
