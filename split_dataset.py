import pandas as pd
from sklearn.model_selection import train_test_split


file_path = 'dataset.csv'
data = pd.read_csv(file_path)


X = data.iloc[:, :-1] #all rows and all columns except the last one
y = data.iloc[:, -1]  #all rows and only the last column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) #70% train data, 30% test data


train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)


train_data.to_csv('train_dataset.csv', index=False)
test_data.to_csv('test_dataset.csv', index=False)

