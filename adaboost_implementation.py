# Adaboost implementation
from os import uname_result
from typing import List
import pandas as pd
import numpy as np
import math
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from random import seed
from logistic_regression_implementation import logistic_regression, predict


def normalize(w):
    temp = w[:]
    s = float(sum(temp))

    for i in range(0, len(temp)):
        temp[i] /= s
    return temp


class AdaBoost:
    def __init__(self, examples_X, examples_Y, K):
        self.examples_X = examples_X
        self.examples_Y = examples_Y
        self.K = K
        self.N = examples_X.shape[0]
        self.w = []

        for i in range(0, self.N):
            self.w.append(1/float(self.N))

        self.h = []
        self.z = []

    def train(self):

        examples_X = self.examples_X
        examples_Y = self.examples_Y
        K = self.K
        w = self.w
        h = self.h
        z = self.z
        N = self.N

        print("Training started")

        for k in range(0, K):
            examples_X.reset_index(inplace=True, drop=True)
            examples_Y.reset_index(inplace=True, drop=True)
            data_X = examples_X.sample(n=round(N/10), replace=False, weights=w)
            data_Y = examples_Y[data_X.index]
            print(data_X.shape)
            print(data_Y.shape)

            l_rate = 0.1
            n_epoch = 20
            error_threshold = 0.0

            yhat, coef = logistic_regression(
                data_X, examples_X, data_Y, l_rate, n_epoch, error_threshold)
            h.append(coef)
            print('Logistic regression finished')

            error = 0

            for j in range(0, N):
                xj = examples_X.iloc[j]
                yj = examples_Y.iloc[j]
                res = round(predict(xj.tolist(), h[k]))

                if res != yj:
                    error += w[j]
            print('Error: ', error)
            if error > 0.5:
                z.append(0)
                continue

            for j in range(0, N):
                xj = examples_X.iloc[j]
                yj = examples_Y.iloc[j]
                res = round(predict(xj.tolist(), h[k]))

                if res == yj:
                    w[j] *= error/(1 - error)

            w = normalize(w)
            z.append(math.log((1 - error)/error, 2))
            print(k, 'th booster finished')

        self.h = h
        self.z = z

    def predict_using_adaboost(self, attributes):
        h = self.h
        z = self.z
        s = 0
        for k in range(0, self.K):
            res = round(predict(attributes.tolist(), h[k]))
            if res == 0:
                res = -1
            s += (res*z[k])
        if s >= 0:
            return 1
        return 0


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
        X_train = dataset.drop(['exceeds'], axis=1)
        y_train = dataset['exceeds']

    K = 5

    booster = AdaBoost(X_train, y_train, K)
    booster.train()

    y_train = y_train.tolist()
    y_pred_t = list()
    for i in range(0, len(y_train)):
        y_pred_t.append(booster.predict_using_adaboost(X_train.iloc[i]))
    accuracy = accuracy_score(y_train, y_pred_t)
    print('Accuracy: %f' % accuracy)

    y_test = y_test.tolist()
    y_pred = list()
    for i in range(0, len(y_test)):
        y_pred.append(booster.predict_using_adaboost(X_test.iloc[i]))
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy: %f' % accuracy)


if __name__ == "__main__":
    main()
