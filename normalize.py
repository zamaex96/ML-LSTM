import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, Normalizer

# Read the dataset
file_path = 'dataset.csv'
data = pd.read_csv(file_path)

# Split the dataset into features and target
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target

# Initialize scalers
scalers = {
     #This method scales the data to a fixed range, usually 0 to 1.
    'MinMaxScaler': MinMaxScaler(),
    'StandardScaler': StandardScaler(),
    #This method is robust to outliers by using the median and the interquartile range for scaling.
    'RobustScaler': RobustScaler(),
    #This method scales each feature by its maximum absolute value, preserving the sparsity of the data.
    'MaxAbsScaler': MaxAbsScaler(),
     #This method transforms features to follow a uniform or normal distribution. It's useful when the data does not follow a Gaussian distribution.
    'QuantileTransformer': QuantileTransformer(output_distribution='normal'),
    #This method scales individual samples to have unit norm (i.e., the sum of the squares is 1). It's useful for text classification and clustering.
    'Normalizer': Normalizer()
}

# Apply each scaler
normalized_data = {}
for scaler_name, scaler in scalers.items():
    # Also known as standardization, this method centers the data to have a mean of 0 and a standard deviation of 1.
    X_scaled = scaler.fit_transform(X)
    X_normalized = pd.DataFrame(X_scaled, columns=X.columns)
    normalized_data[scaler_name] = pd.concat([X_normalized, y], axis=1)

# Display normalized data for each method
for scaler_name, data in normalized_data.items():
    print(f"Normalized Data using {scaler_name}:\n", data.head(), "\n")

