from operator import index
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize, KBinsDiscretizer

# Load the CSV file
filename = 'creditcard.csv'
data = pd.read_csv(filename)

print(data['Class'].value_counts())

# Exploratory data analysis
# print(data.head())
# print(data.dtypes)

# Seperate dataframe for positive class and negative class
df_credit_pos = data.loc[data.iloc[:, data.shape[1] - 1] == 1]
df_credit_neg = data.loc[data.iloc[:, data.shape[1] - 1] == 0]
# print(df_credit_pos.shape)

# Sample 5k negative class datapoints and concat with the positive class as the dataset is imbalanced
df_credit_neg = df_credit_neg.sample(n=1000, replace=False)
df_credit = pd.concat([df_credit_neg, df_credit_pos], axis=0)
print(df_credit['Class'].value_counts())
df_credit_class = df_credit['Class'].copy()

df_credit = df_credit.drop(['Class'], axis=1)
# print(df_credit.head())

# Handle missing values in the label
# temp = df_credit.isna().sum()
# print(temp) # No missing value in the dataset

# Scale the numeric columns
scaler = StandardScaler().fit(
    df_credit.loc[:, df_credit.columns != 'Class'])
data_scaled = scaler.transform(
    df_credit.loc[:, df_credit.columns != 'Class'])
# print(data_scaled[0:10])
print(df_credit.head())
# Normalize the numeric columns
data_normalized = normalize(data_scaled, norm='l2')
# print(data_normalized[0:10])

# Discretize the numeric columns
est = KBinsDiscretizer(n_bins=20,
                       encode='ordinal').fit(data_normalized)
data_discretized = est.transform(data_normalized)
# data_discretized['Class'] = df_credit['Class']
# print(data_discretized[0:5])

# Converting the data_discretized array to dataframe
df = pd.DataFrame(data_discretized, columns=df_credit.columns)
df['Class'] = df_credit_class.values

print(df['Class'].value_counts())
preprocessed_filename = filename.split('.')[0] + '_preprocessed.csv'
df.to_csv(preprocessed_filename, index=False)
