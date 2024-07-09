Simple LSTM model which is trained on time-series dataset composed of two features and two outputs.

# Summary of Split-dataset

## Overview
This Python script prepares a dataset (`dataset.csv`) for training and testing by splitting it into training and testing sets using scikit-learn's `train_test_split` function. It then saves the split datasets into separate CSV files (`train_dataset.csv` and `test_dataset.csv`).

## Libraries Used
- `pandas`: Data manipulation and analysis library.
- `sklearn.model_selection`: Module from scikit-learn for splitting data into training and testing sets.

## Steps

1. **Read Data**: Reads the dataset from the file path `'dataset.csv'` using `pd.read_csv()`.

2. **Split Data**:
   - Splits the dataset into features (`X`) and target (`y`).
   - Uses `train_test_split(X, y, test_size=0.3, random_state=42)` to split the data:
     - `X_train`, `X_test`: Training and testing features.
     - `y_train`, `y_test`: Training and testing targets.
     - `test_size=0.3`: Sets the proportion of the dataset to include in the test split to 30%.
     - `random_state=42`: Sets the random seed for reproducibility.

3. **Concatenate DataFrames**:
   - Combines training features (`X_train`) with training target (`y_train`) into `train_data` using `pd.concat([X_train, y_train], axis=1)`.
   - Combines testing features (`X_test`) with testing target (`y_test`) into `test_data` using `pd.concat([X_test, y_test], axis=1)`.

4. **Save Data**:
   - Saves `train_data` to `'train_dataset.csv'` and `test_data` to `'test_dataset.csv'` using `to_csv()` with `index=False` to exclude row indices from being saved.

## Notes
- The script assumes the existence of `'dataset.csv'` in the current directory and saves the split datasets accordingly.
- It demonstrates how to split a dataset into training and testing sets for machine learning tasks using scikit-learn and manage data with pandas.



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


