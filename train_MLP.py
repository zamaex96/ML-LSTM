import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import pandas as pd
from model import MLPModel  # Ensure MLPModel is imported from your model.py
from data_loader import CustomDataset  # Ensure CustomDataset is imported

# Define model parameters
input_size = 8  # Number of input features
hidden_size = 11  # Number of hidden units
output_size = 4   # Number of output classes

# Instantiate the MLPModel
model = MLPModel(input_size, hidden_size, output_size)

# Set device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Paths to training and testing data
# Paths to training and testing data
train_csv_path = r"C:\train.csv"
test_csv_path = r"C:\test.csv"

# Create datasets and dataloaders
train_dataset = CustomDataset(train_csv_path)
test_dataset = CustomDataset(test_csv_path)

train_data_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=12, shuffle=False)

# Define number of epochs
epochs = 100
train_loss_values = []
test_loss_values = []
train_accuracy_values = []
test_accuracy_values = []

for epoch in range(epochs):
    # Training phase
    model.train()
    epoch_train_loss = 0.0
    correct_train = 0
    total_train = 0
    for inputs, labels in train_data_loader:
        inputs, labels = inputs.to(device), labels.to(device)

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
        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_train_loss:.4f}, Test Loss: {epoch_test_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')

# Save the model state
torch.save(model.state_dict(), 'saved_model_mlp.pth')

# Create a DataFrame to save training info
train_info = {'train_loss': train_loss_values,
              'train_accuracy': train_accuracy_values,
              'test_loss': test_loss_values,
              'test_accuracy': test_accuracy_values}

train_info_df = pd.DataFrame(train_info)
train_info_df.to_csv("train_loss_mlp.csv", index=False)

# Plot the loss and accuracy on the same figure
plt.figure(figsize=(12, 4))

# Loss plot
plt.subplot(2, 1, 1)
plt.plot(range(1, epochs + 1), train_loss_values, label='Training Loss')
plt.plot(range(1, epochs + 1), test_loss_values, label='Testing Loss')
plt.title('Training and Testing Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy plot
plt.subplot(2, 1, 2)
plt.plot(range(1, epochs + 1), train_accuracy_values, label='Training Accuracy')
plt.plot(range(1, epochs + 1), test_accuracy_values, label='Testing Accuracy')
plt.title('Training and Testing Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.show()
