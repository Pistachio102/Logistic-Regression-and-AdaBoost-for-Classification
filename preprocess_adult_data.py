from operator import index
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize, KBinsDiscretizer

# Load the CSV file
filename = 'adult_data.csv'
data = pd.read_csv(filename)
data.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'exceeds', ]

# Exploratory data analysis
# print(data.head())

# Handle missing values in the label
rows_to_be_dropped = []
temp = data.iloc[:, data.shape[1] - 1].isnull()
for j in range(0, data.shape[0]):
    if temp.iloc[j] is True:
        rows_to_be_dropped.append(j)
        print('Row ' + str(j) + ' dropped')
data = data.drop(rows_to_be_dropped)
data = data.reset_index(drop=True)


# Handle missing values in the features
categorical_columns_minus_label = [1, 3, 5, 6, 7, 8, 9, 13]
for j in range(0, data.shape[1] - 1):
    if j in categorical_columns_minus_label:
        temp = data.iloc[:, j].value_counts().idxmax()
        data.iloc[:, j] = data.iloc[:, j].replace(np.nan, temp, regex=True)
    else:
        temp = data.iloc[:, j].mean()
        data.iloc[:, j] = data.iloc[:, j].replace(np.nan, temp, regex=True)

# Convert categorical columns to numerical columns
categorical_columns_minus_label = [1, 3, 5, 6, 7, 8, 9, 13, 14]
for j in categorical_columns_minus_label:
    en = LabelEncoder()
    en.fit(data.iloc[:, j])
    data.iloc[:, j] = en.transform(data.iloc[:, j])

preprocessed_filename = filename.split('.')[0] + '_preprocessed.csv'
data.to_csv(preprocessed_filename, index=False)
print(data.head())
