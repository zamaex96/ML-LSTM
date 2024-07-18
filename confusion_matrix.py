import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from model import *
from  data_loader import *

# Function to compute the confusion matrix
def plot_confusion_matrix(true_labels, predicted_labels, class_names):
    cm = confusion_matrix(true_labels, predicted_labels)
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

input_size = 2  #number of features
hidden_size = 4   # layers
output_size = 2    # number of classes

model_inference = LSTMModel(input_size, hidden_size, output_size)
model_inference.load_state_dict(torch.load('saved_model_lstm.pth',map_location=torch.device('cpu'))) # "saved_model_lstm" is the pre-trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_inference.to(device)

test_csv_path = 'test_dataset.csv'
test_dataset = CustomDataset(test_csv_path)
test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model_inference.eval()
true_labels = []
predicted_labels = []

with torch.no_grad():
    for inputs, labels in test_data_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model_inference(inputs)
        _, predicted = torch.max(outputs.data, 1)

        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())


class_names = ["X", 'Y']

plot_confusion_matrix(true_labels, predicted_labels, class_names)

