Simple LSTM model which is trained on time-series dataset composed of two features and two outputs.

# Summary of Train.py

## Overview
This Python script trains and evaluates a neural network model (LSTM) using PyTorch on a custom dataset. It includes data loading, model definition, training loop, evaluation, and visualization of training and testing metrics (loss and accuracy).

## Libraries Used
- `torch`: Deep learning framework for tensor computations.
- `torch.nn`: Neural network module to define and train neural network models.
- `torch.optim`: Optimization algorithms like SGD for updating model parameters.
- `torch.utils.data`: Tools for data loading and handling.
- `matplotlib.pyplot`: Plotting library for visualization.
- `pandas`: Data manipulation and analysis library.

## Model Initialization
- Defines the neural network model (`LSTMModel`) with input size, hidden size, and output size.
- Moves the model to GPU (`cuda`) if available, otherwise uses CPU.

## Loss Function and Optimizer
- Uses Cross Entropy Loss (`nn.CrossEntropyLoss()`) as the loss criterion for classification tasks.
- Uses Stochastic Gradient Descent (`optim.SGD`) as the optimizer to update model parameters.

## Dataset Handling
- Loads training and testing datasets (`train_dataset.csv` and `test_dataset.csv`) using a custom dataset class (`CustomDataset`).
- Creates data loaders (`train_data_loader` and `test_data_loader`) for efficient batch-wise data processing.

## Training Loop
- Iterates over a specified number of epochs (100 epochs).
- Sets the model to training mode (`model.train()`), computes training loss and accuracy.
- Updates model parameters using backpropagation (`loss.backward()`) and optimizer (`optimizer.step()`).
- Tracks and stores training loss and accuracy metrics.

## Testing Loop
- Sets the model to evaluation mode (`model.eval()`), computes testing loss and accuracy.
- Uses `torch.no_grad()` to disable gradient calculation during testing to conserve memory and speed up computation.
- Tracks and stores testing loss and accuracy metrics.

## Model Saving
- Saves the trained model's state dictionary to a file (`saved_model_lstm.pth`) after training completes.

## Results Visualization
- Generates plots showing the training and testing loss over epochs (`train_loss_lstm.csv`).
- Plots the training and testing accuracy over epochs.

## Execution
- The script prints periodic updates of training and testing metrics every 5 epochs during training.

## Notes
- The code assumes the availability of CUDA-enabled GPU for acceleration if `torch.cuda.is_available()` evaluates to `True`.
- It demonstrates typical steps in training and evaluating a neural network model using PyTorch, including data loading, model definition, training loop, evaluation, and result visualization.


