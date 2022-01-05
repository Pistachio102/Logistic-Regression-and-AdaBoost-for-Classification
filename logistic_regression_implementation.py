# Logistic Regression
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from random import seed
from random import randrange
from csv import reader
from math import exp

# Compute impurity of a feature


def compute_impurity(feature):

    probs = feature.value_counts(normalize=True)
    # Compute impurity based on entropy
    impurity = -1 * np.sum(np.log2(probs) * probs)

    return(round(impurity, 3))

# Compute information gain


def compute_feature_information_gain(df, target, feature):

    target_entropy = compute_impurity(target)

    entropy_list = list()
    weight_list = list()

    for level in df[feature].unique():
        df_feature_level = df[df[feature] == level]
        entropy_level = compute_impurity(target)
        entropy_list.append(round(entropy_level, 3))
        weight_level = len(df_feature_level) / len(df)
        weight_list.append(round(weight_level, 3))

    feature_remaining_impurity = np.sum(
        np.array(entropy_list) * np.array(weight_list))

    information_gain = target_entropy - feature_remaining_impurity

    return(information_gain)


# Return specific number of important features based on the information gain
def get_feature_list(df, target, number_of_features):
    feature_list = []

    for feature in df.columns:
        feature_info_gain = compute_feature_information_gain(
            df, target, feature)
        feature_list.append([feature, feature_info_gain])
    # First column is the serial number of the rows. So starting from the second column
    n_feature_list = sorted(feature_list, key=lambda l: l[1], reverse=True)[
        0:number_of_features]

    return n_feature_list

# Split a dataset into k folds


def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Calculate accuracy percentage


def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        print(actual[i], "  ", predicted[i])
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split


def evaluate_algorithm(X_t, y_t, n_folds, *args):
    kf = KFold(n_splits=5, random_state=1, shuffle=True)
    scores = list()
    coefficients = list()
    for train_index, test_index in kf.split(X_t):
        X_train_k, X_test_k = X_t.iloc[train_index], X_t.iloc[test_index]
        y_train_k, y_test_k = y_t.iloc[train_index], y_t.iloc[test_index]
        predicted, coefficients = logistic_regression(
            X_train_k, X_test_k, y_train_k, *args)
        actual = y_test_k.to_list()
        # accuracy = accuracy_metric(actual, predicted)
        accuracy = accuracy_score(actual, predicted)
        print('Accuracy: %f' % accuracy)
        # precision tp / (tp + fp)
        precision = precision_score(actual, predicted)
        print('Precision: %f' % precision)
        # recall: tp / (tp + fn)
        recall = recall_score(actual, predicted)
        print('Recall: %f' % recall)
        # specificity and false discovery rate
        tn, fp, fn, tp = confusion_matrix(actual, predicted).ravel()
        specificity = tn / (tn+fp)
        print('Specificity: %f' % specificity)
        false_discovery_rate = fp / (fp+tp)
        print('False discovery rate: %f' % false_discovery_rate)
        # f1: 2 tp / (2 tp + fp + fn)
        f1 = f1_score(actual, predicted)
        print('F1 score: %f' % f1)
        scores.append([accuracy, recall, specificity,
                      precision, false_discovery_rate, f1])
        print("K-fold finished")
    scores_np = np.array(scores)
    mean_scores = scores_np.mean(axis=0)
    return mean_scores.tolist(), coefficients


# Make a prediction with coefficients
def predict(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row)):
        yhat += coefficients[i + 1] * row[i]
    if yhat < -20:
        return 0

    return 1.0 / (1.0 + exp(-yhat))
    # return ( exp(yhat) - exp(-yhat)) / ( exp(yhat) + exp(-yhat))

# Estimate logistic regression coefficients using gradient descent


def coefficients_gd(x_train_k, y_train_k, l_rate, n_epoch, error_threshold):
    number_of_coeffs = len(x_train_k.columns) + 1
    coef = [0.0 for i in range(number_of_coeffs)]

    for epoch in range(n_epoch):
        for index, row in x_train_k.iterrows():
            yhat = predict(row, coef)
            error = y_train_k[index] - yhat
            coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
            for i in range(len(row)):
                coef[i + 1] = coef[i + 1] + l_rate * \
                    error * yhat * (1.0 - yhat) * row[i]
            if error_threshold == 0:
                continue
            if error < error_threshold:
                return coef

    return coef

# Linear Regression Algorithm With Stochastic Gradient Descent


def logistic_regression(x_train_k, x_test_k, y_train_k, l_rate, n_epoch, error_threshold):
    predictions = list()
    coef = coefficients_gd(x_train_k, y_train_k, l_rate,
                           n_epoch, error_threshold)
    for index, row in x_test_k.iterrows():
        yhat = predict(row, coef)
        yhat = round(yhat)
        predictions.append(yhat)
    return predictions, coef


def main():
    seed(1)
    # Load data
    # filename = 'Telco-Customer-Churn_preprocessed.csv'
    # filename = 'creditcard_preprocessed.csv'
    filename = 'adult_data_preprocessed.csv'
    dataset = pd.read_csv(filename)

    # Split the data into train and test set
    if filename == "Telco-Customer-Churn_preprocessed.csv":
        X = dataset.drop(['Churn'], axis=1)
        y = dataset['Churn']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=1)

    elif filename == "creditcard_preprocessed.csv":
        X = dataset.drop(['Class'], axis=1)
        y = dataset['Class']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=1)

    elif filename == "adult_data_preprocessed.csv":
        X = dataset.drop(['exceeds'], axis=1)
        y = dataset['exceeds']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=1)

    number_of_features = 6

    # Get list of desired number of features based on information gain
    list_of_features = get_feature_list(X_train, y_train, number_of_features)

    # New dataframe of the wanted columns
    name_of_features = [feature[0] for feature in list_of_features]
    selected_columns = X_train[name_of_features]
    df = selected_columns.copy()

    # print(df.head())

    # Evaluation using Logistic Regression
    # evaluate algorithm
    n_folds = 5
    l_rate = 0.1
    n_epoch = 100
    error_threshold = 0.0
    scores, coefficients = evaluate_algorithm(
        df, y_train, n_folds, l_rate, n_epoch, error_threshold)
    print('Scores on Train Set: ', scores)
    # print(df.dtypes)

    # print(list_of_features)
    # Test
    selected_columns = X_test[name_of_features]
    test_x = selected_columns.copy()
    predicted_y = list()
    for index, row in test_x.iterrows():
        print('row: ', len(row))
        print('coeff: ', len(coefficients))
        yhat = predict(row, coefficients)
        yhat = round(yhat)
        predicted_y.append(yhat)
    actual = y_test.to_list()
    # accuracy = accuracy_metric(actual, predicted)
    accuracy = accuracy_score(actual, predicted_y)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(actual, predicted_y)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(actual, predicted_y)
    print('Recall: %f' % recall)
    # specificity and false discovery rate
    tn, fp, fn, tp = confusion_matrix(actual, predicted_y).ravel()
    specificity = tn / (tn+fp)
    print('Specificity: %f' % specificity)
    false_discovery_rate = fp / (fp+tp)
    print('False discovery rate: %f' % false_discovery_rate)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(actual, predicted_y)
    print('F1 score: %f' % f1)


if __name__ == "__main__":
    main()
