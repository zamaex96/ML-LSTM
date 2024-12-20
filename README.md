

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

This Python script performs inference using a trained LSTM, CNN, RNN, and MLP model (`LSTMModel`) to predict a class label based on input data.

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


# Summary of Train_LSTM.py

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

# Training and Evaluation Summary of train_CNN

This code trains and evaluates a **1D Convolutional Neural Network (1D-CNN)** for a classification task with 17 output classes using a time series dataset. The process involves data loading, model training, evaluation, and result visualization.

---

## Key Components

### Model and Training Setup
- **Model:** `CNN1DModel` - A custom 1D-CNN model with:
  - Input size: `200` features.
  - Output size: `17` classes.
- **Loss Function:** CrossEntropyLoss.
- **Optimizer:** Stochastic Gradient Descent (SGD) with a learning rate of `0.001`.
- **Device:** GPU if available; otherwise, CPU.

### Data
- **Training Data:** Loaded from a CSV file (`train.csv`).
- **Testing Data:** Loaded from a CSV file (`test.csv`).
- **Custom Dataset Class:** `CustomDataset`, handles loading and preprocessing.

### Training Configuration
- **Epochs:** `1000`.
- **Batch Size:** `12`.

---

## Workflow

### Training Loop
1. **Forward Pass:** Inputs are passed through the model.
2. **Loss Calculation:** Loss is computed using CrossEntropyLoss.
3. **Backward Pass:** Gradients are calculated and used to update model parameters.
4. **Metrics Calculation:** Accuracy is computed for training and testing data.

### Evaluation Loop
- The model is evaluated on test data after each epoch to compute:
  - Testing loss.
  - Testing accuracy.

### Output Generation
1. **Model Saving:** The trained model is saved in `.pth` format.
2. **Training Logs:** Loss and accuracy values are saved as a CSV file.
3. **Visualization:**
   - Loss vs. Epochs (Training and Testing).
   - Accuracy vs. Epochs (Training and Testing).
   - Plots are saved as both `.png` and `.pdf`.

---

## Output Locations
- **Model Directory:** `C:\Users\BU\Documents\BULabProjects\DUI Detection\ML\Models`
- **CSV Directory:** `C:\Users\BU\Documents\BULabProjects\DUI Detection\ML\CSV`
- **Plots Directory:** `C:\Users\BU\Documents\BULabProjects\DUI Detection\ML\Plots`

---

## Visualization
Two plots are generated to visualize the performance of the model:
1. **Loss Plot:** Displays the training and testing loss over epochs.
2. **Accuracy Plot:** Shows training and testing accuracy over epochs.

The plots are saved in high-resolution PNG and PDF formats for better readability.

---

## Example Output
```plaintext
Epoch [5/1000], Train Loss: 1.2345, Test Loss: 1.4567, Train Accuracy: 85.67%, Test Accuracy: 84.23%
Model saved at C:\Users\BU\Documents\BULabProjects\DUI Detection\ML\Models\1D-CNN_TS.pth
Training data saved at C:\Users\BU\Documents\BULabProjects\DUI Detection\ML\CSV\1D-CNN_TS.csv
Plots saved at C:\Users\BU\Documents\BULabProjects\DUI Detection\ML\Plots\1D-CNN_TS.png and C:\Users\BU\Documents\BULabProjects\DUI Detection\ML\Plots\1D-CNN_TS.pdf
```
# Training and Evaluation Summary for MLP Model

This code trains and evaluates a **Multi-Layer Perceptron (MLP)** model for classification, leveraging PyTorch. The process includes data preprocessing, model training, testing, saving results, and visualizing performance.

---

## Key Components

### Model Details
- **Model:** `MLPModel`
  - Input size: `12` features.
  - Hidden size: `11` units.
  - Output size: `17` classes.
- **Device:** GPU (if available) or CPU.

### Training Configuration
- **Loss Function:** CrossEntropyLoss.
- **Optimizer:** Stochastic Gradient Descent (SGD) with a learning rate of `0.001`.
- **Epochs:** `100`.
- **Batch Size:** `12`.

### Data
- **Training Data:** Loaded from `train.csv`.
- **Testing Data:** Loaded from `test.csv`.
- **Custom Dataset Class:** `CustomDataset` (handles data loading and preprocessing).

---

## Workflow

### Training Phase
1. Forward pass through the MLP model.
2. Compute the loss using CrossEntropyLoss.
3. Perform backpropagation to calculate gradients.
4. Update model parameters using SGD.
5. Track training loss and accuracy.

### Testing Phase
- Evaluate the model on test data after every epoch.
- Compute test loss and accuracy without updating model weights.

### Output
1. **Model State:** Saved as `saved_model_mlp.pth`.
2. **Training Logs:** Loss and accuracy values saved as `train_loss_mlp.csv`.
3. **Visualization:**
   - Loss vs. Epochs (Training and Testing).
   - Accuracy vs. Epochs (Training and Testing).

---

## Visualization
The code generates a plot with two subplots:
1. **Loss Plot:** Training and testing loss over epochs.
2. **Accuracy Plot:** Training and testing accuracy over epochs.

The plots provide insights into model performance during training.

---

## Example Output
```plaintext
Epoch [5/100], Train Loss: 0.5678, Test Loss: 0.6543, Train Accuracy: 85.67%, Test Accuracy: 84.12%
Model saved at saved_model_mlp.pth
Training data saved at train_loss_mlp.csv
```
# Training and Evaluation Summary for RNN Model

This code trains and evaluates an **RNN-based model** for multi-class classification using PyTorch. It includes loading datasets, training the model, evaluating performance, saving outputs, and visualizing results.

---

## Key Components

### Model Details
- **Model:** `RNNModel`
  - Input size: `12` features.
  - Hidden size: `11` units.
  - Number of layers: `8`.
  - Output size: `17` classes.
- **Device:** GPU (if available) or CPU.

### Training Configuration
- **Loss Function:** CrossEntropyLoss.
- **Optimizer:** Adam, learning rate `0.001`.
- **Epochs:** `100`.
- **Batch Size:** `12`.

### Data
- **Training Data:** `train.csv`.
- **Testing Data:** `test.csv`.
- **Dataset Class:** `CustomDataset` (handles data loading and preprocessing).

---

## Workflow

### Training Phase
1. Forward pass through the RNN model.
2. Compute loss using CrossEntropyLoss.
3. Backpropagation to compute gradients.
4. Update parameters using the Adam optimizer.
5. Track training loss and accuracy.

### Testing Phase
- Evaluate the model after each epoch without updating parameters.
- Compute test loss and accuracy.

---

## Outputs

1. **Model Checkpoint:** Saved as a `.pth` file including:
   - Model state dictionary.
   - Hyperparameters such as input size, hidden size, learning rate, etc.
   - Path: `C:\Models\RNN_TSN.pth`.

2. **Training Logs:** 
   - Loss and accuracy values saved as a CSV file.
   - Path: `C:\CSV\RNN_TSN.csv`.

3. **Visualization:**
   - **Loss Plot:** Training and testing loss over epochs.
   - **Accuracy Plot:** Training and testing accuracy over epochs.
   - Plots saved as:
     - PNG: `C:\Plots\RNN_TSN.png`.
     - PDF: `C:\Plots\RNN_TSN.pdf`.

---

## Example Output
```plaintext
Epoch [10/100], Train Loss: 0.1234, Train Accuracy: 92.50%, Test Loss: 0.1345, Test Accuracy: 91.00%
Model saved at C:\Models\RNN_TSN.pth
Training data saved at C:\CSV\RNN_TSN.csv
Plots saved at C:\Plots\RNN_TSN.png and C:\Plots\RNN_TSN.pdf
```
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

#### **1. RNN Model (`RNNModel`)**

- **Purpose**: Designed for sequential data, capturing temporal relationships.
- **Key Components**:
  - **Input Dimensions**:
    - Input can be either 2D (`batch_size, input_size`) or 3D (`batch_size, seq_length, input_size`).
    - Automatically adds a time-step dimension if input is 2D.
  - **Layers**:
    - `nn.RNN`: Recurrent layer for processing sequences.
    - `nn.Linear`: Fully connected layer for producing outputs.
  - **Forward Propagation**:
    - Takes the last hidden state for classification (sequence to label task).

#### **2. 1D CNN Model (`CNN1DModel`)**

- **Purpose**: Designed for extracting features from temporal or 1D spatial data using convolutional layers.
- **Key Components**:
  - **Input Dimensions**:
    - Input must be 3D (`batch_size, channels, input_length`), where `channels=1` for single feature per timestep.
  - **Layers**:
    - **Convolutional Layers**:
      - Two layers (`Conv1d`) with kernel size 3 and padding 1 for feature extraction.
    - **Pooling**:
      - Max pooling (`MaxPool1d`) after each convolution layer to reduce dimensions.
    - **Fully Connected Layers**:
      - `fc1`: Reduces flattened feature size to 128.
      - `fc2`: Maps to the number of output classes.
  - **Forward Propagation**:
    - Extracts features via convolution and pooling.
    - Flattens features and applies fully connected layers for output.

---

### Comparison

| **Aspect**         | **RNN Model**                          | **1D CNN Model**                    |
|---------------------|----------------------------------------|-------------------------------------|
| **Primary Use**     | Sequential or temporal data            | Feature extraction in 1D signals    |
| **Key Layers**      | RNN, Linear                            | Conv1d, MaxPool1d, Linear           |
| **Handling Sequence** | Processes sequentially, maintains temporal order | Uses convolution to capture local patterns |
| **Output**          | Uses the last hidden state for classification | Flattened feature vector processed by FC layers |

---

### Implementation Notes

- For **RNNModel**, input should be 2D (`batch_size, input_size`) or 3D (`batch_size, seq_length, input_size`).
- For **CNN1DModel**, ensure input shape is `(batch_size, 1, input_length)`.
- Both models can handle batch processing for efficient training.
  
### Alternatives to Cross-Entropy Loss and Optimizers:

#### **1. Alternatives to Cross-Entropy Loss**
Cross-entropy loss is widely used for classification problems, but other loss functions may be suitable based on your task:

- **Focal Loss**
  - Suitable for imbalanced datasets.
  - Focuses more on hard-to-classify samples by down-weighting easy samples.
  - Implementation: Available in libraries like PyTorch or custom implementations.

- **Mean Squared Error (MSE)**
  - Typically used for regression but can be applied to classification when one-hot encoding is used.
  - Not ideal for classification as it treats probabilities linearly.

- **Kullback-Leibler Divergence Loss (KLDivLoss)**
  - Measures the divergence between two probability distributions.
  - Useful when comparing soft labels or probabilistic outputs.

- **Hinge Loss**
  - Commonly used for binary classification tasks with Support Vector Machines (SVMs).
  - Encourages a margin of separation between classes.

- **Label Smoothing**
  - A variation of cross-entropy loss that smooths target labels to prevent overconfidence.
  - Useful in cases prone to overfitting or noisy labels.

- **Binary Cross-Entropy (BCE)**
  - Specialized for binary classification tasks.
  - Can also be extended to multi-label classification problems.

- **Contrastive Loss**
  - Useful in tasks like face recognition or similarity learning.
  - Operates on pairs of samples to measure the similarity or dissimilarity.

---

#### **2. Alternatives to Stochastic Gradient Descent (SGD) Optimizer**
Depending on the nature of your problem and dataset, alternative optimizers may provide better convergence:

- **Adam (Adaptive Moment Estimation)**
  - Combines the advantages of RMSProp and momentum.
  - Well-suited for sparse data and non-stationary objectives.
  - Common usage: `optim.Adam(model.parameters(), lr=0.001)`

- **AdamW (Adam with Weight Decay Regularization)**
  - Variation of Adam with improved weight decay regularization.
  - Helps prevent overfitting.
  - Common usage: `optim.AdamW(model.parameters(), lr=0.001)`

- **RMSProp (Root Mean Square Propagation)**
  - Divides the learning rate by a running average of the magnitudes of recent gradients.
  - Well-suited for recurrent neural networks (RNNs).
  - Common usage: `optim.RMSprop(model.parameters(), lr=0.001)`

- **Adagrad (Adaptive Gradient Algorithm)**
  - Adapts learning rates based on historical gradient information.
  - Suitable for sparse data or parameters.
  - Common usage: `optim.Adagrad(model.parameters(), lr=0.001)`

- **Adadelta**
  - Addresses some limitations of Adagrad by restricting step size.
  - Common usage: `optim.Adadelta(model.parameters(), lr=1.0)`

- **NAdam (Nesterov-accelerated Adam)**
  - Extends Adam by incorporating Nesterov momentum.
  - Common usage: `optim.NAdam(model.parameters(), lr=0.001)`

- **LBFGS (Limited-memory BFGS)**
  - A quasi-Newton method optimizer.
  - Suitable for smaller datasets and optimization problems with second-order behavior.
  - Common usage: `optim.LBFGS(model.parameters(), lr=0.1)`

---

### Selecting Alternatives:
- **Classification Tasks**: 
  - Use Focal Loss or Label Smoothing if data is imbalanced.
  - Use Hinge Loss for binary classification with margin-based separation.

- **Optimizers for Stability**:
  - Adam and AdamW are generally more stable for deep learning tasks.
  - RMSProp is preferred for RNNs or non-stationary datasets.

- **Fine-Tuning Hyperparameters**:
  - Experiment with learning rates, momentum, and weight decay to adapt optimizers to your dataset.

### Example:
```python
# Alternative Loss and Optimizer
criterion = nn.CrossEntropyLoss()  # Or FocalLoss(), KLDivLoss(), etc.
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)  # Or RMSProp, Adagrad
```
  
### Table of Alternatives to Cross-Entropy Loss and SGD Optimizer

| **Type**               | **Name**              | **Description**                                                                                                                                   | **Implementation (PyTorch)**                                                                                     |
|------------------------|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| **Loss Functions**     | **Cross-Entropy Loss** | Default for classification tasks.                                                                                                                 | `nn.CrossEntropyLoss()`                                                                                          |
|                        | **Focal Loss**         | Focuses on hard-to-classify samples; reduces the influence of easy samples.                                                                       | [Focal Loss Implementation](https://github.com/AdeelH/pytorch-multi-class-focal-loss)                            |
|                        | **Mean Squared Error** | Regression-based loss, less common for classification.                                                                                           | `nn.MSELoss()`                                                                                                   |
|                        | **KL Divergence Loss** | Measures the divergence between predicted and target distributions.                                                                               | `nn.KLDivLoss()`                                                                                                 |
|                        | **Hinge Loss**         | Encourages a margin of separation between classes; used in SVMs.                                                                                  | `nn.HingeEmbeddingLoss()`                                                                                        |
|                        | **Label Smoothing**    | Reduces overconfidence by smoothing target labels.                                                                                                | `nn.CrossEntropyLoss(label_smoothing=0.1)` (PyTorch 1.10+)                                                       |
|                        | **Binary Cross-Entropy** | For binary or multi-label classification.                                                                                                        | `nn.BCELoss()` or `nn.BCEWithLogitsLoss()`                                                                       |
|                        | **Contrastive Loss**   | Used for similarity or metric learning tasks.                                                                                                    | Custom: See [Contrastive Loss Implementation](https://omoindrot.github.io/triplet-loss)                          |
| **Optimizers**         | **SGD**               | Basic optimizer with momentum.                                                                                                                    | `optim.SGD(model.parameters(), lr=0.01, momentum=0.9)`                                                           |
|                        | **Adam**              | Combines RMSProp and momentum; adapts learning rates.                                                                                            | `optim.Adam(model.parameters(), lr=0.001)`                                                                       |
|                        | **AdamW**             | Adam with decoupled weight decay for better regularization.                                                                                       | `optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)`                                                  |
|                        | **RMSProp**           | Scales learning rates based on recent gradient magnitudes; good for RNNs.                                                                        | `optim.RMSprop(model.parameters(), lr=0.001)`                                                                    |
|                        | **Adagrad**           | Adapts learning rates for parameters with infrequent updates.                                                                                    | `optim.Adagrad(model.parameters(), lr=0.01)`                                                                     |
|                        | **Adadelta**          | Improves Adagrad by limiting step sizes for better stability.                                                                                     | `optim.Adadelta(model.parameters(), lr=1.0)`                                                                     |
|                        | **NAdam**             | Combines Adam and Nesterov momentum for faster convergence.                                                                                      | `optim.NAdam(model.parameters(), lr=0.001)`                                                                      |
|                        | **LBFGS**             | Quasi-Newton method for small datasets or second-order optimization.                                                                              | `optim.LBFGS(model.parameters(), lr=0.1)`                                                                        |

### Example Code Snippet

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Example Loss Function: Cross-Entropy Loss
criterion = nn.CrossEntropyLoss()

# Example Optimizer: AdamW
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Focal Loss Example (Custom Implementation)
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss
        return F_loss

criterion = FocalLoss(alpha=0.25, gamma=2)
```

### Notes
- Use **Cross-Entropy Loss** for most classification tasks, unless specific challenges like class imbalance or noisy labels exist.
- Use **AdamW** or **RMSProp** as alternatives to SGD for better convergence in deep learning tasks.
  
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

### Explanation of Metrics Calculation and Saving

The provided code evaluates model performance using various metrics and saves the results to a text file for documentation purposes. Below is a detailed breakdown of the script:

#### **Accuracy Calculation**
- **Overall Accuracy**: Percentage of correctly predicted samples over the total number of samples.
  ```python
  accuracy = 100 * correct / total
  print(f"\nOverall Accuracy: {accuracy:.2f}%")
  ```

- **Per-Class Accuracy**: Calculates the accuracy for each class using the confusion matrix.
  ```python
  cm = confusion_matrix(true_labels, predicted_labels)
  for i, class_name in enumerate(class_names):
      class_correct = cm[i, i]
      class_total = cm[i].sum()
      class_acc = class_correct / class_total * 100
      print(f"{class_name}: {class_acc:.2f}%")
  ```

#### **Metrics Calculation**
Metrics like precision, recall, and F1-score are computed using **scikit-learn**:
- **Accuracy**: The ratio of correctly predicted samples over total samples.
- **Precision**: The ability of the model to avoid false positives, calculated per class and averaged (macro).
- **Recall**: The ability of the model to find all relevant samples, calculated per class and averaged (macro).
- **F1-Score**: The harmonic mean of precision and recall.
  ```python
  accuracy = accuracy_score(true_labels, predicted_labels) * 100
  precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=0) * 100
  recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=0) * 100
  f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=0) * 100
  ```

#### **Saving Metrics to a Text File**
- The script saves all calculated metrics and detailed per-class accuracy to a timestamped file.
- It ensures the output directory exists and creates the file with the following content:
  ```python
  os.makedirs(output_folder2, exist_ok=True)
  metrics_filename = os.path.join(output_folder2, f"model_metrics_{model_name}_{ext}_{timestamp}.txt")

  with open(metrics_filename, 'w') as f:
      f.write(f"Model: {model_name}\n")
      f.write(f"Detail: {ext}\n")
      f.write(f"Number of Classes: {output_size}\n\n")
      f.write(f"Overall Accuracy: {accuracy:.2f}%\n")
      f.write(f"Precision (macro): {precision:.2f}%\n")
      f.write(f"Recall (macro): {recall:.2f}%\n")
      f.write(f"F1 Score (macro): {f1:.2f}%\n\n")
      f.write("Per-class Accuracy:\n")
      for i, class_name in enumerate(class_names):
          class_correct = cm[i, i]
          class_total = cm[i].sum()
          class_acc = class_correct / class_total * 100
          f.write(f"{class_name}: {class_acc:.2f}%\n")
  ```

#### **Metrics File Example**
The saved text file will look like this:
```
Model: MyModel
Detail: Experiment_01
Number of Classes: 5

Overall Accuracy: 92.30%
Precision (macro): 90.50%
Recall (macro): 91.80%
F1 Score (macro): 91.00%

Per-class Accuracy:
Class_A: 95.00%
Class_B: 89.50%
Class_C: 92.00%
Class_D: 90.00%
Class_E: 93.50%
```

#### **File Management**
- Ensures files are uniquely identified with a timestamp for traceability.
- The metrics are stored in the specified directory, making it easy to organize results from multiple experiments.
  
## Output

- Displays the accuracy score in percentage.
- Shows a heatmap of the confusion matrix with predicted versus true labels.

## Notes

- Ensure the `LSTMModel` and `CustomDataset` classes are correctly defined in their respective files (`model.py` and `data_loader.py`).
- Adjust `class_names` and other parameters as per specific dataset classes and requirements.

# ModelLoader
### Explanation of the Modules

#### 1. **InceptionBlock Class**
The `InceptionBlock` implements a single block from the Inception architecture, but adapted for 1D convolutions. It consists of parallel convolutional paths with different kernel sizes and a max-pooling branch, followed by batch normalization, ReLU activation, and dropout.

- **Parallel Convolution Layers**: 
  - There are 3 parallel convolution layers, each with a different kernel size (`9`, `19`, and `39`). These layers are designed to capture different spatial features in the input data, with varying receptive fields.
  - **Padding** is applied based on kernel size (`kernel_size // 2`) to maintain the input length.
  - **BatchNorm1d** normalizes the output of each convolution layer.
  - **ReLU Activation** introduces non-linearity.
  - **Dropout** is applied to avoid overfitting by randomly zeroing some of the activations during training.

- **MaxPool Branch**:
  - A 1D max-pooling operation with kernel size `3` and stride `1` is applied. It helps reduce the dimensionality of the data while retaining important features.
  - A 1D convolution with a kernel size of `1` is applied to reduce the number of channels in the output, followed by batch normalization, ReLU, and dropout.

- **Concatenation**:
  - The outputs of the parallel convolutions and the max-pooling branch are concatenated along the channel dimension. This is a key feature of the Inception architecture, which merges multiple types of feature extraction.

- **Batch Normalization, ReLU, and Dropout**:
  - After concatenating the outputs, the resulting tensor is passed through a batch normalization layer to normalize the features, followed by a ReLU activation function and a dropout layer for regularization.

---

#### 2. **InceptionTime Class**
The `InceptionTime` class implements the complete model with multiple Inception blocks for time-series classification, utilizing multiple inception blocks with residual connections, global average pooling, and a fully connected classifier.

- **Initial Convolution Layer (`conv1`)**:
  - This is the first convolutional layer applied to the input time-series data, transforming it into a feature map. The kernel size is `1`, which does not modify the temporal dimension, but it learns the spatial features across the input channels.
  - It is followed by batch normalization, ReLU activation, and dropout for regularization.

- **Inception Blocks (`inception_blocks`)**:
  - The `InceptionTime` model consists of multiple `InceptionBlock` layers, organized in a list. The number of inception blocks is determined by the `num_blocks` parameter.
  - **Residual Connections**: Every third inception block is followed by a shortcut connection, adding the input of that block to the output. This helps preserve the initial features and avoids vanishing gradients by allowing gradients to flow directly through the shortcut connections.

- **Global Average Pooling**:
  - After the inception blocks, a **Global Average Pooling** layer is used, which reduces the feature map from `(batch_size, channels, length)` to `(batch_size, channels)` by averaging the values in the temporal dimension. This helps reduce the spatial dimension and serves as a form of dimensionality reduction before the final classification layer.

- **Fully Connected Classifier (`classifier`)**:
  - The output of the global average pooling is passed through a fully connected layer, which maps the features to a lower-dimensional space (using `inception_channels // 2`).
  - The classifier consists of:
    - **Linear Layer**: A fully connected layer that reduces the number of features from `inception_channels` to `inception_channels // 2`.
    - **BatchNorm1d**: Normalization layer for the output.
    - **ReLU Activation**: Non-linearity.
    - **Dropout**: Regularization to reduce overfitting.
    - **Final Linear Layer**: This layer maps the output to the number of classes (`num_classes`), producing the final classification scores.

- **Shortcut Connections**:
  - The model uses shortcut connections every third inception block. A shortcut is created by passing the output of the previous layer through a convolutional layer (`Conv1d`) to match the output size of the current block. The result is added back to the current block output, allowing the model to maintain important features and improve gradient flow during training.

- **Weight Initialization**:
  - The `apply` method is used to apply custom weight initialization to the layers:
    - **Convolutional layers**: Initialized using Kaiming (He) initialization, which is effective for layers with ReLU activation.
    - **Batch Normalization layers**: Initialized with `1` for weights and `0` for biases.
    - **Linear layers**: Initialized using Kaiming initialization for weights and `0` for biases.

---

#### 3. **CustomDataset Class**
The `CustomDataset` class is a subclass of `torch.utils.data.Dataset` and is used to handle data loading from a CSV file for training. It preprocesses the data into the format expected by PyTorch models.

- **Initialization**:
  - The `__init__` method reads the CSV file using `pandas.read_csv` and splits the data into features and labels. The features are stored in `self.features`, and the labels in `self.labels`. Each row of the dataset represents a single data point, with the last column being the label.

- **Length**:
  - The `__len__` method returns the total number of samples in the dataset. This is required by PyTorch's `DataLoader` for batching the data.

- **Get Item**:
  - The `__getitem__` method fetches a single sample by index `idx`. It converts the features and labels from numpy arrays to PyTorch tensors:
    - **Features**: Converted into a `FloatTensor` to ensure they are treated as continuous data (e.g., float values).
    - **Label**: Converted into a `LongTensor` to represent integer labels, which are typically used for classification tasks.

---

### Summary
- The **InceptionBlock** uses multiple parallel convolution layers with different kernel sizes and a max-pooling branch to extract features at different scales. It concatenates their outputs and applies batch normalization, activation, and dropout.
- **InceptionTime** stacks multiple **InceptionBlocks** and incorporates shortcut connections for better gradient flow and feature preservation, followed by global average pooling and a fully connected classifier.
- **CustomDataset** facilitates loading time-series data from a CSV file into PyTorch by converting it into tensors suitable for training.

# Train Inception Time Model
This code demonstrates how to train and evaluate an `InceptionTime` model for time-series classification using PyTorch. Below is a step-by-step explanation of the entire process:

---

### 1. **Model Parameters and Setup**
```python
model_name = "InceptionTime"
ext = "TS"

# Model parameters
input_size = 10001  # number of features/columns
num_blocks = 6  # Number of InceptionTime blocks
output_size = 9  # labels/target variables
batch_Size = 32
num_epochs = 200
learning_Rate = 0.01
```
- **Model Parameters**: These parameters define the architecture and training configuration.
  - `input_size`: Number of input features per sample (10001 features in this case).
  - `num_blocks`: Number of Inception blocks in the `InceptionTime` model.
  - `output_size`: Number of classes (9 classes for classification).
  - `batch_Size`: Number of samples processed together in one training step (32 in this case).
  - `num_epochs`: Total number of epochs (200 epochs).
  - `learning_Rate`: The learning rate for the optimizer (0.01).

---

### 2. **InceptionTime Model Initialization**
```python
model = InceptionTime(input_size, output_size, num_blocks)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")
```
- **InceptionTime Model**: The `InceptionTime` model is instantiated using the provided parameters (`input_size`, `output_size`, and `num_blocks`).
- **Device Selection**: The model is moved to the available computing device (`cuda` for GPU or `cpu` for CPU).
- **Model to Device**: `model.to(device)` moves the model to the selected device.

---

### 3. **Loss Function and Optimizer**
```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_Rate,
    weight_decay=0.001
)
```
- **Loss Function**: `CrossEntropyLoss` is used as the criterion because it is suitable for multi-class classification.
- **Optimizer**: The `AdamW` optimizer is used, which is an adaptive learning rate optimizer with weight decay (L2 regularization).
  - `learning_Rate`: The learning rate for the optimizer (0.01).
  - `weight_decay`: Regularization to prevent overfitting.

---

### 4. **Learning Rate Scheduler**
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.1,
    patience=10,
    min_lr=1e-6
)
```
- **Learning Rate Scheduler**: The `ReduceLROnPlateau` scheduler reduces the learning rate by a factor of 0.1 if the validation loss does not improve for `patience` epochs. The minimum learning rate is set to `1e-6`.

---

### 5. **Dataset Loading**
```python
train_csv_path = r"C:\train.csv"
test_csv_path = r"C:\test.csv"

train_dataset = CustomDataset(train_csv_path)
test_dataset = CustomDataset(test_csv_path)

train_data_loader = DataLoader(train_dataset, batch_size=batch_Size, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=batch_Size, shuffle=False)
```
- **Paths**: The paths to the training and testing CSV files are specified (`train_csv_path` and `test_csv_path`).
- **CustomDataset**: A custom `Dataset` class (`CustomDataset`) is used to load the CSV data into a PyTorch-compatible format. The dataset is expected to load both features and labels.
- **DataLoader**: The `DataLoader` is used to load the dataset in batches (`batch_Size = 32`), with the training data being shuffled.

---

### 6. **Training Loop**
```python
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
```
- **Training Phase**:
  - **Model Training Mode**: The model is set to training mode using `model.train()`.
  - **Loss and Accuracy Calculation**: For each batch in the `train_data_loader`:
    - Inputs and labels are moved to the device (GPU/CPU).
    - **Forward Pass**: The model computes the output from the input.
    - **Loss Calculation**: The loss is calculated using the `CrossEntropyLoss` criterion.
    - **Backward Pass**: Gradients are computed and the optimizer updates the model's parameters.
    - **Accuracy Calculation**: The predicted labels are compared with the true labels to calculate accuracy.

---

### 7. **Testing Loop**
```python
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
```
- **Testing Phase**:
  - **Model Evaluation Mode**: The model is set to evaluation mode using `model.eval()`.
  - **No Gradient Calculation**: `torch.no_grad()` disables gradient calculation, speeding up inference.
  - **Loss and Accuracy Calculation**: Similar to the training phase, but no gradients are computed, and the model’s parameters remain unchanged.

---

### 8. **Saving the Model, Training Logs, and Plots**
```python
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
```
- **Saving Model**: The trained model’s state dictionary and hyperparameters are saved using `torch.save` to a `.pth` file at the specified path.

```python
train_info = {'train_loss': train_loss_values,
              'train_accuracy': train_accuracy_values,
              'test_loss': test_loss_values,
              'test_accuracy': test_accuracy_values}

csv_path = os.path.join(output_folder2, f"{model_name}_{ext}.csv")
train_info_df = pd.DataFrame(train_info)
train_info_df.to_csv(csv_path, index=False)
```
- **Training Logs**: Training and testing losses and accuracies are saved to a CSV file for later analysis.

```python
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
```
- **Plotting Loss and Accuracy**: The training and testing loss, as well as accuracy, are plotted using `matplotlib` and saved as PNG and PDF files.

---

### 9. **Output Summary**
- The model, training logs, and plots are saved to specified directories for further analysis and future use.
- **Training Progress**: Every 5 epochs, the loss and accuracy for both training and testing are printed.



<div align="center">
  <a href="https://maazsalman.org/">
    <img width="70" src="click-svgrepo-com.svg" alt="gh" />
  </a>
  <p> Explore More! 🚀</p>
</div>



