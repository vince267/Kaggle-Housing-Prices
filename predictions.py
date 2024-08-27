# import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error 
# from sklearn.preprocessing import OneHotEncoder

# File path for training data and validation data
train_file_path = '../Kaggle-Housing-Prices/home-data-for-ml-course/train.csv'
test_file_path = '../Kaggle-Housing-Prices/home-data-for-ml-course/test.csv'

# Store training and validation data
training_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# Store target column (sale price) as y 
y = training_data.SalePrice

# List of features we will focus on
features = ['LotArea', 'OverallQual', 'FullBath', 'BedroomAbvGr']

# string_features = ['RoofMatl', 'RoofStyle']

# Store columns with features as X 
X = training_data[features]

# Store test data as test_X
test_X = test_data[features]

# Split X and y into train and validation data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# enc = OneHotEncoder()
# enc.fit(train_X)
# train_X = enc.transform(train_X).toarray()
# training_data.columns = le.transform(training_data.columns)
# features = le.transform(features)
# print(training_data.columns)


# Specify model
rf_model = RandomForestRegressor(random_state=1)

# Fit model to training data
rf_model.fit(train_X, train_y)

# # Make predictions about validation data
predictions = rf_model.predict(val_X)
id_list = test_data.Id 

# # Compare predictions with actual sale price
MAE = mean_absolute_error(predictions, val_y)

print('The MAE is', MAE)

submission_predictions = rf_model.predict(test_X)
csv_data = [['id', 'SalePrice']]
for i in range(len(id_list)):
    csv_data.append([id_list[i], submission_predictions[i]])

data_frame = pd.DataFrame(csv_data)
data_frame.to_csv('submission.csv', header=False, index=False)
# with open('submission.csv', 'w', newline='') as csvfile:
#     submission_writer = csv.writer(csvfile)
#     submission_writer.writerow(csv_data)
