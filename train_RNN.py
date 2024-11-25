import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from data_loader import CustomDataset  # Assuming CustomDataset loads your data
from model import RNNModel


model_name = "RNN"
ext="TSN"

# Model parameters
input_size = 12
hidden_size = 11
output_size = 17
num_layers = 8
num_epochs=100
learning_Rate=0.001
batch_Size=12
# Instantiate RNN model
model = RNNModel(input_size, hidden_size, num_layers, output_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_Rate)  # Using Adam optimizer

# Paths to training and testing data
train_csv_path = r"C:\test.csv"
test_csv_path = r"C:\test.csv"

# Assuming CustomDataset class loads your data
train_dataset = CustomDataset(train_csv_path)
test_dataset = CustomDataset(test_csv_path)

train_data_loader = DataLoader(train_dataset, batch_size=batch_Size, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=batch_Size, shuffle=False)

epochs = num_epochs
train_loss_values = []
test_loss_values = []
train_accuracy_values = []
test_accuracy_values = []

# Assuming you have a DataLoader `train_data_loader` and `test_data_loader` for your datasets
for epoch in range(epochs):
    # Training phase
    model.train()
    epoch_train_loss = 0.0
    correct_train = 0
    total_train = 0
    for inputs, labels in train_data_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Ensure outputs and labels have the same batch size
        if outputs.size(0) != labels.size(0):
            raise ValueError(
                f'Expected input batch_size ({outputs.size(0)}) to match target batch_size ({labels.size(0)})')

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()

        # Calculate accuracy
        _, predicted_train = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()

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

            # Forward pass
            outputs = model(inputs)

            # Ensure outputs and labels have the same batch size
            if outputs.size(0) != labels.size(0):
                raise ValueError(
                    f'Expected input batch_size ({outputs.size(0)}) to match target batch_size ({labels.size(0)})')

            # Calculate loss
            loss = criterion(outputs, labels)
            epoch_test_loss += loss.item()

            # Calculate accuracy
            _, predicted_test = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted_test == labels).sum().item()

    epoch_test_loss /= len(test_data_loader)
    test_loss_values.append(epoch_test_loss)
    test_accuracy = 100 * correct_test / total_test
    test_accuracy_values.append(test_accuracy)

    # Print epoch results
    print(
        f'Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Loss: {epoch_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')


# Specify output folder for saving the model, CSV file, and plots
output_folder1 = r"C:\Models"# Replace with desired path
os.makedirs(output_folder1, exist_ok=True)  # Create folder if it doesn't exist
output_folder2 = r"C:\CSV"  # Replace with desired path
os.makedirs(output_folder2, exist_ok=True)  # Create folder if it doesn't exist
output_folder3 = r"C:\Plots"  # Replace with desired path
os.makedirs(output_folder3, exist_ok=True)  # Create folder if it doesn't exist
# Save the model state
# Save the final model

model_path = os.path.join(output_folder1, f"{model_name}_{ext}.pth")
torch.save({
    'model_state_dict': model.state_dict(),
    'input_size': input_size,
    'hidden_size': hidden_size,
    'num_epochs': num_epochs,
    'output_size': output_size,
    'num_layers': num_layers,
    'learning_rate': learning_Rate,
    'hyperparameters': {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_epochs': num_epochs,
        'output_size': output_size,
        'num_layers': num_layers,
        'learning_rate': learning_Rate,
         'batch_size': batch_Size,
    }
}, model_path)
print(f"Model saved at {model_path}")

# Plotting loss and accuracy
# (Code for plotting remains the same as in the previous response)
train_info = {'train_loss': train_loss_values,
              'train_accuracy': train_accuracy_values,
              'test_loss': test_loss_values,
              'test_accuracy': test_accuracy_values}

train_info_df = pd.DataFrame(train_info)
csv_path = os.path.join(output_folder2, f"{model_name}_{ext}.csv")
train_info_df.to_csv(csv_path, index=False)
print(f"Training data saved at {csv_path}")

# Plotting loss and accuracy
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
# Save plots as PNG and PDF
png_file_path = os.path.join(output_folder3, f"{model_name}_{ext}.png")
pdf_file_path = os.path.join(output_folder3, f"{model_name}_{ext}.pdf")
plt.savefig(png_file_path, format='png', dpi=600)
plt.savefig(pdf_file_path, format='pdf', dpi=600)
print(f"Plots saved at {png_file_path} and {pdf_file_path}")
plt.tight_layout()
plt.show()
