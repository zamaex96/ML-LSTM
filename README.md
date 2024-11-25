

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

<div align="center">
  <a href="https://maazsalman.org/">
    <img width="70" src="click-svgrepo-com.svg" alt="gh" />
  </a>
  <p> Explore More! 🚀</p>
</div>



