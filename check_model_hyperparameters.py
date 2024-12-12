import torch

# Path to the saved model
model_path = r"C:\Models\HybridCNNLSTM.pth"

# Load the checkpoint
checkpoint = torch.load(model_path,weights_only=True)

# Check if hyperparameters were saved along with the model
if 'hyperparameters' in checkpoint:
    hyperparams = checkpoint['hyperparameters']
    print("Saved Hyperparameters:")
    for key, value in hyperparams.items():
        print(f"{key}: {value}")
else:
    print("No hyperparameters were found in the checkpoint.")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(model_path, map_location=device,weights_only=True)
print(checkpoint.keys())  # Check the keys to see if it matches expected  keys
