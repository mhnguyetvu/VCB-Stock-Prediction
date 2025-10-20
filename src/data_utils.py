import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(train_path, test_path):
    dataset_train = pd.read_csv(train_path)
    training_set = dataset_train.iloc[:, 1:2].values
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    X_train, y_train = [], []
    no_of_sample = len(training_set)
    for i in range(60, no_of_sample):
        X_train.append(training_set_scaled[i-60:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    dataset_test = pd.read_csv(test_path)
    real_stock_price = dataset_test.iloc[:, 1:2].values
    dataset_total = pd.concat((dataset_train['CLOSE'], dataset_test['CLOSE']), axis=0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)
    return X_train, y_train, sc, dataset_test, real_stock_price, inputs, dataset_train, dataset_total
