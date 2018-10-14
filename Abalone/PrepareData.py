import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def prepare_dataset():
    # read in dataset
    data = pd.read_csv(os.path.join(os.getcwd(), 'abalone-data.txt'),
                  names=['sex', 'length', 'diameter', 'height', 'whole_weight',
                        'shucked_weight', 'viscera_weight', 'shell_weight', 'rings'])

    # one-hot encode categorical features
    data['sex'] = LabelEncoder().fit_transform(data['sex'])
    data = pd.get_dummies(data, columns=['sex'])

    # create Train, Validation, and Test sets
    X = data.drop('rings', axis=1)
    y = data['rings']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=3)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=3)

    # standardize continuous values to zero mean and unit variance
    scaler = StandardScaler()
    X_train[['length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight']] = scaler.fit_transform(
        X_train[['length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight']])

    X_val[['length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight']] = scaler.transform(
        X_val[['length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight']])
    X_test[['length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight']] = scaler.transform(
        X_test[['length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight']])

    return X_train, X_val, X_test, y_train, y_val, y_test