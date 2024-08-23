import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error 

# File path for training data and validation data
train_file_path = '../Kaggle-Housing-Prices/home-data-for-ml-course/train.csv'
test_file_path = '../Kaggle-Housing-Prices/home-data-for-ml-course/test.csv'

# Store training and validation data
training_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# Store target column (sale price) as y 
y = training_data.SalePrice

# List of features we will focus on
features = ['LotArea', 'OverallQual']

# Store columns with features as X 
X = training_data[features]

# Store test data as test_X
test_X = test_data[features]

# Split X and y into train and validation data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Specify model
rf_model = RandomForestRegressor(random_state=1)

# Fit model to training data
rf_model.fit(train_X, train_y)

# # Make predictions about validation data
predictions = rf_model.predict(val_X)

# # Compare predictions with actual sale price
MAE = mean_absolute_error(predictions, val_y)

print(MAE)