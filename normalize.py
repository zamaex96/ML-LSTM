import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Read the dataset
file_path = 'dataset.csv'
data = pd.read_csv(file_path)

# Split the dataset into features and target
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the features
X_scaled = scaler.fit_transform(X)

# Convert the scaled features back to a DataFrame
X_normalized = pd.DataFrame(X_scaled, columns=X.columns)

# Combine the normalized features with the target variable
data_normalized = pd.concat([X_normalized, y], axis=1)

# Display the normalized data
print(data_normalized.head())
