import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import pandas as pd
import os
from ModelLoader import *
model_name = "InceptionTime"
ext = "TS"

# Model parameters
input_size = 10001 # number of features/columns
num_blocks = 6  # Number of InceptionTime blocks
output_size = 9 # labels/target variables
batch_Size = 32
num_epochs = 200
learning_Rate = 0.01

# Define InceptionTime Model
model = InceptionTime(input_size, output_size, num_blocks)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(model.parameters(), lr=learning_Rate)  # Recommended optimizer for InceptionTime
# Learning rate scheduler
# Modified optimizer with weight decay
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_Rate,
    weight_decay=0.001
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.1,
    patience=10,
    min_lr=1e-6
)

# Paths to training and testing data
train_csv_path = r"C:\train.csv"
test_csv_path = r"C:\test.csv"

train_dataset = CustomDataset(train_csv_path)
test_dataset = CustomDataset(test_csv_path)

train_data_loader = DataLoader(train_dataset, batch_size=batch_Size, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=batch_Size, shuffle=False)

# Training loop
train_loss_values = []
test_loss_values = []
train_accuracy_values = []
test_accuracy_values = []

for epoch in range(num_epochs):
    # Training phase
    model.train()
    epoch_train_loss = 0.0
    correct_train = 0
    total_train = 0
    for inputs, labels in train_data_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()

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

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            epoch_test_loss += loss.item()

            _, predicted_test = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted_test == labels).sum().item()

    epoch_test_loss /= len(test_data_loader)
    test_loss_values.append(epoch_test_loss)
    test_accuracy = 100 * correct_test / total_test
    test_accuracy_values.append(test_accuracy)
    scheduler.step(epoch_test_loss)
    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {epoch_train_loss:.4f}, '
              f'Test Loss: {epoch_test_loss:.4f}, '
              f'Train Accuracy: {train_accuracy:.2f}%, '
              f'Test Accuracy: {test_accuracy:.2f}%')

# Save the model, training logs, and plots
output_folder1 = r"C:\ML\Models"
output_folder2 = r"C:\ML\CSV"
output_folder3 = r"C:\ML\Plots"

os.makedirs(output_folder1, exist_ok=True)
os.makedirs(output_folder2, exist_ok=True)
os.makedirs(output_folder3, exist_ok=True)

model_path = os.path.join(output_folder1, f"{model_name}_{ext}.pth")
torch.save({
    'model_state_dict': model.state_dict(),
    'hyperparameters': {
        'input_size': input_size,
        'num_blocks': num_blocks,
        'num_epochs': num_epochs,
        'output_size': output_size,
        'learning_rate': learning_Rate,
        'batch_size': batch_Size,
    }
}, model_path)
print(f"Model saved at {model_path}")

train_info = {'train_loss': train_loss_values,
              'train_accuracy': train_accuracy_values,
              'test_loss': test_loss_values,
              'test_accuracy': test_accuracy_values}

csv_path = os.path.join(output_folder2, f"{model_name}_{ext}.csv")
train_info_df = pd.DataFrame(train_info)
train_info_df.to_csv(csv_path, index=False)
print(f"Training data saved at {csv_path}")

# Plot loss and accuracy
plt.figure(figsize=(12, 4))
plt.subplot(2, 1, 1)
plt.plot(range(1, num_epochs + 1), train_loss_values, label='Training Loss')
plt.plot(range(1, num_epochs + 1), test_loss_values, label='Testing Loss')
plt.title('Training and Testing Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(range(1, num_epochs + 1), train_accuracy_values, label='Training Accuracy')
plt.plot(range(1, num_epochs + 1), test_accuracy_values, label='Testing Accuracy')
plt.title('Training and Testing Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

png_path = os.path.join(output_folder3, f"{model_name}_{ext}.png")
pdf_path = os.path.join(output_folder3, f"{model_name}_{ext}.pdf")
plt.savefig(png_path, format='png', dpi=600)
plt.savefig(pdf_path, format='pdf', dpi=600)
print(f"Plots saved at {png_path} and {pdf_path}")
plt.tight_layout()
plt.show()
