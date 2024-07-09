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

# Summary of DataLoader

## Overview

This Python code defines a custom dataset class (`CustomDataset`) using PyTorch's `Dataset` class. The dataset is initialized with data from a CSV file (`csv_file`). Each data sample consists of two features and one class label.

## Libraries Used

- `torch`: PyTorch library for tensor computations and neural networks.
- `torch.nn`: PyTorch's neural network module.
- `torch.optim`: PyTorch's optimization module.
- `torch.utils.data`: PyTorch's data loading utilities.
- `matplotlib.pyplot`: Matplotlib for plotting.
- `pandas`: Data manipulation library.

## Custom Dataset Class (`CustomDataset`)

### Initialization (`__init__` method)
- **Parameters**: 
  - `csv_file`: Path to the CSV file containing the dataset.
  - `transform`: Optional transformation to be applied on features (e.g., data augmentation).

- **Functionality**:
  - Loads the dataset from `csv_file` using `pd.read_csv()` into `self.data`.
  - Initializes with an optional `transform` parameter to preprocess features.

### Length (`__len__` method)
- **Functionality**:
  - Returns the number of samples in the dataset (`len(self.data)`).

### Get Item (`__getitem__` method)
- **Parameters**:
  - `idx`: Index of the sample to retrieve.

- **Functionality**:
  - Retrieves features (`features`) and class label (`class_name`) for the sample at index `idx`.
  - Converts features into a PyTorch tensor (`torch.tensor`) of type `torch.float32`.
  - Converts class label into a PyTorch tensor of type `torch.long`.
  - Applies an optional transformation (`self.transform`) on features if provided.

- **Returns**:
  - A tuple containing `features` and `class_name`.

## Usage
- The `CustomDataset` class is used to encapsulate and preprocess data from a CSV file for machine learning tasks in PyTorch.
- Suitable for tasks like supervised learning where data needs to be loaded, preprocessed, and transformed into PyTorch tensors.

## Notes
- Ensure the CSV file (`csv_file`) has the appropriate format and is accessible to the script.
- The dataset class facilitates data loading, transformation, and indexing, essential for training neural networks in PyTorch.

# Summary of  Model Inference

## Overview

This Python script performs inference using a trained LSTM model (`LSTMModel`) to predict a class label based on input data.

## Libraries Used

- `torch`: PyTorch library for tensor computations and neural networks.
- `model`: Assumes the existence of a module `model` containing `LSTMModel`.
- `torch.cuda`: PyTorch module for CUDA support (if available).
- `torch.tensor`: Constructs PyTorch tensors.
- `torch.no_grad`: Context manager to disable gradient computation for inference.
- `print`: Standard output function for displaying results.

## Model Inference Setup

### Model Initialization
- **Parameters**: 
  - `input_size`: Dimensionality of input features.
  - `hidden_size`: Number of units in the LSTM hidden state.
  - `output_size`: Number of output classes.

- **Functionality**:
  - Initializes an instance of `LSTMModel` for inference with specified `input_size`, `hidden_size`, and `output_size`.
  - Loads the model's trained weights (`saved_model_lstm.pth`) using `torch.load()` and `model_inference.load_state_dict()`.

### Device Configuration
- **Device Selection**: 
  - Determines the device (GPU or CPU) available using `torch.cuda.is_available()`.
  - Moves the model (`model_inference`) to the selected device using `.to(device)`.

### Input Data Preparation
- **Test Input**: 
  - Prepares a test input tensor (`test_input`) containing a single sample (`[[150, 19]]`) of input features, converted to `torch.float32` and moved to the selected device (`device`).

### Model Evaluation
- **Evaluation Mode**: 
  - Sets the model to evaluation mode using `model_inference.eval()`.
  - Disables gradient computation during inference using `torch.no_grad()`.

### Inference and Prediction
- **Forward Pass**: 
  - Performs a forward pass through the model with the test input (`test_input`) to obtain predicted probabilities (`predicted_probs`) for each class.
  - Computes the predicted class label (`predicted_labels`) by selecting the class with the highest probability using `torch.max()`.

### Output
- **Print Statement**: 
  - Displays the predicted class label (`predicted_labels.item()`) as the output of the inference process.

## Usage
- This script demonstrates how to load a pretrained LSTM model, perform inference on a single input sample, and obtain the predicted class label.
- Suitable for applications requiring predictive modeling with sequential data using LSTM networks in PyTorch.

## Notes
- Ensure the existence of the `model` module with `LSTMModel` implemented and compatible with the provided input and output sizes.
- Adjust `test_input` according to the expected input format of the LSTM model (`input_size` should match the number of features).
  
This summary provides an overview of how the provided Python script performs inference using a pretrained LSTM model in PyTorch, including model initialization, input data preparation, model evaluation, and prediction.


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

  
# Summary of Models

## Overview

This Python script defines two different models using PyTorch's `nn.Module`:

1. **MLPModel**:
   - Multilayer Perceptron model with two fully connected (linear) layers followed by ReLU activation and softmax output.
   - Suitable for classification tasks where the input data does not have a sequential relationship.

2. **LSTMModel**:
   - Long Short-Term Memory (LSTM) model consisting of an LSTM layer followed by a linear layer and softmax output.
   - Designed for sequential data where the order of input elements matters, such as time series or natural language processing tasks.

## Libraries Used

- `torch`: PyTorch library for tensor computations and neural networks.
- `torch.nn`: PyTorch's neural network module for defining layers and models.
- `torch.optim`: PyTorch module for optimization algorithms.
- `torch.utils.data`: PyTorch utilities for handling datasets and data loading.
- `matplotlib.pyplot`: Library for plotting graphs and visualizations.
- `pandas`: Library for data manipulation and analysis.

## Models Defined

### MLPModel

- **Constructor (`__init__`)**:
  - Initializes two fully connected layers (`fc1`, `fc2`) with ReLU activation (`relu`) and softmax activation (`softmax`).

- **Forward Method (`forward`)**:
  - Defines the forward pass of the model:
    - Applies the first linear transformation (`fc1`).
    - Applies ReLU activation.
    - Applies the second linear transformation (`fc2`).
    - Applies softmax activation to output probabilities across classes.

### LSTMModel

- **Constructor (`__init__`)**:
  - Initializes an LSTM layer (`lstm`) with `batch_first=True` to accept input tensors with batch size as the first dimension.
  - Defines a fully connected layer (`fc`) and softmax activation (`softmax`).

- **Forward Method (`forward`)**:
  - Defines the forward pass of the LSTM model:
    - Processes the input tensor (`x`) through the LSTM layer (`lstm`).
    - Applies a linear transformation (`fc`) to the LSTM output.
    - Applies softmax activation to obtain class probabilities.

## Usage

- Use `MLPModel` for non-sequential data tasks such as basic classification where order does not matter.
- Use `LSTMModel` for sequential data tasks such as time series prediction or natural language processing where the order of input elements is important.

## Notes

- Adjust `input_size`, `hidden_size`, and `output_size` parameters according to the specific requirements of your task and input data dimensions.
- Ensure compatibility of input data shapes (`batch_size`, `sequence_length`, `feature_dimensions`) with the defined models.
- These models assume standard classification outputs; adjust `output_size` for tasks with different numbers of classes.

# Summary of Confusion Matrix

## Overview

This Python script performs inference using an LSTM model (`LSTMModel`) trained on a saved model (`saved_model_lstm.pth`). It then computes and plots a confusion matrix and accuracy score using the test dataset (`test_dataset.csv`).

## Libraries Used

- `torch`: PyTorch library for tensor computations and neural networks.
- `torch.nn`: PyTorch's neural network module for defining layers and models.
- `torch.utils.data`: PyTorch utilities for handling datasets and data loading.
- `pandas`: Library for data manipulation and analysis.
- `seaborn`: Statistical data visualization library based on Matplotlib.
- `matplotlib.pyplot`: Library for plotting graphs and visualizations.
- `sklearn.metrics`: Scikit-learn library for performance metrics such as confusion matrix and accuracy score.

## Models and Data Loading

- **LSTMModel**: Defined in the `model.py` file, loaded from `saved_model_lstm.pth`.
- **CustomDataset**: Defined in `data_loader.py`, loads the test dataset (`test_dataset.csv`) using `torch.utils.data.Dataset` and `torch.utils.data.DataLoader`.

## Functionality

1. **Loading Model**:
   - Loads the `LSTMModel` from a saved state dictionary (`saved_model_lstm.pth`) onto the CPU or GPU based on availability.

2. **Loading Test Dataset**:
   - Uses `CustomDataset` to load the test dataset (`test_dataset.csv`) into a `DataLoader` for batch processing.

3. **Inference**:
   - Sets the model to evaluation mode (`model_inference.eval()`).
   - Iterates over batches of data from `test_data_loader`, performs forward pass through the model, and collects predictions.

4. **Confusion Matrix and Accuracy**:
   - Computes the confusion matrix and accuracy score using `sklearn.metrics.confusion_matrix` and `sklearn.metrics.accuracy_score`.
   - Plots the confusion matrix using `seaborn.heatmap` with annotated values and class labels (`["bad", "good"]`).

## Output

- Displays the accuracy score in percentage.
- Shows a heatmap of the confusion matrix with predicted versus true labels.

## Notes

- Ensure the `LSTMModel` and `CustomDataset` classes are correctly defined in their respective files (`model.py` and `data_loader.py`).
- Adjust `class_names` and other parameters as per specific dataset classes and requirements.



