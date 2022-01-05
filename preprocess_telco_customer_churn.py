import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize, KBinsDiscretizer


def load_and_preprocess_dataset(filename):
    # Load the CSV file
    data = pd.read_csv(filename)

    # Drop "customerID" column
    data = data.drop(['customerID'], axis=1)

    # Replace empty strings in "TotalCharges" column with 0 and convert its datatype to float
    data["TotalCharges"] = data["TotalCharges"].replace(
        r'^\s*$', 0, regex=True)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"])

    # Scale the numeric columns
    scaler = StandardScaler().fit(
        data[['tenure', 'MonthlyCharges', 'TotalCharges']])
    data_scaled = scaler.transform(
        data[['tenure', 'MonthlyCharges', 'TotalCharges']])

    # Normalize the numeric columns
    data_normalized = normalize(data_scaled, norm='l2')

    # Discretize the numeric columns
    est = KBinsDiscretizer(n_bins=[20, 20, 20],
                           encode='ordinal').fit(data_normalized)
    data_discretized = est.transform(data_normalized)

    # Replace the processed columns with the respective main dataframe columns
    data["tenure"] = data_discretized[:, 0].tolist()
    data["MonthlyCharges"] = data_discretized[:, 1].tolist()
    data["TotalCharges"] = data_discretized[:, 2].tolist()

    # Encode the categorical columns
    data['gender'] = LabelEncoder().fit_transform(data['gender'])
    data['Partner'] = LabelEncoder().fit_transform(data['Partner'])
    data['Dependents'] = LabelEncoder().fit_transform(data['Dependents'])
    data['PhoneService'] = LabelEncoder().fit_transform(data['PhoneService'])

    data['MultipleLines'] = LabelEncoder().fit_transform(data['MultipleLines'])
    data['InternetService'] = LabelEncoder(
    ).fit_transform(data['InternetService'])
    data['OnlineSecurity'] = LabelEncoder(
    ).fit_transform(data['OnlineSecurity'])
    data['OnlineBackup'] = LabelEncoder().fit_transform(data['OnlineBackup'])

    data['DeviceProtection'] = LabelEncoder(
    ).fit_transform(data['DeviceProtection'])
    data['TechSupport'] = LabelEncoder().fit_transform(data['TechSupport'])
    data['StreamingTV'] = LabelEncoder().fit_transform(data['StreamingTV'])
    data['StreamingMovies'] = LabelEncoder(
    ).fit_transform(data['StreamingMovies'])

    data['Contract'] = LabelEncoder().fit_transform(data['Contract'])
    data['PaperlessBilling'] = LabelEncoder(
    ).fit_transform(data['PaperlessBilling'])
    data['PaymentMethod'] = LabelEncoder().fit_transform(data['PaymentMethod'])
    data['Churn'] = LabelEncoder().fit_transform(data['Churn'])
    print(data["Churn"].value_counts())
    # print(data[0:10])
    preprocessed_filename = filename.split('.')[0] + '_preprocessed.csv'
    data.to_csv(preprocessed_filename, index=False)


y = load_and_preprocess_dataset('Telco-Customer-Churn.csv')
