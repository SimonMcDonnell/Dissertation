import numpy as np
np.random.seed(42)
import random as rn
rn.seed(12345)
import pandas as pd
import os
import tensorflow as tf
tf.set_random_seed(82)
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def prepare_abalone():
    # read in dataset
    data = pd.read_csv(os.path.join(os.getcwd(), 'Abalone/abalone-data.txt'),
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

    return X_train.values, X_val.values, X_test.values, y_train, y_val, y_test


def prepare_concrete():
    # read in dataset
    data = pd.read_csv(os.path.join(os.getcwd(), 'Concrete/concrete_data.csv'))

    # create Train, Validation, and Test sets
    X = data.drop('ccs', axis=1)
    y = data['ccs']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.11, random_state=1)

    # standardize continuous values to zero mean and unit variance
    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_val = X_scaler.transform(X_val)
    X_test = X_scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


def prepare_bank():
    # read in dataset
    data = pd.read_csv(os.path.join(os.getcwd(), 'BankNotes/banknote_authentication.txt'), 
        names=['variance', 'skewness', 'curtosis', 'entropy', 'class'])

    # create Train, Validation, and Test sets
    X = data.drop('class', axis=1)
    y = data['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.11, random_state=1)

    # standardize continuous values to zero mean and unit variance
    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_val = X_scaler.transform(X_val)
    X_test = X_scaler.transform(X_test)
    y_train = y_train.values.reshape(-1, 1)
    y_val = y_val.values.reshape(-1,)
    y_test = y_test.values.reshape(-1,)

    return X_train, X_val, X_test, y_train, y_val, y_test


def prepare_iris():
    # read in dataset
    data = pd.read_csv(os.path.join(os.getcwd(), 'Iris/iris.data'), names=['sepal_length', 'sepal_width', 'petal_length',
                                                                  'petal_width', 'class'])
                                                        
    # create Train, Validation, and Test sets
    X = data.drop('class', axis=1)
    y = data['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.16, random_state=3)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.19, random_state=3)

    # standardize continuous values to zero mean and unit variance, one-hot encode outputs
    X_scaler = StandardScaler()
    y_encoder = LabelEncoder()
    X_train = X_scaler.fit_transform(X_train)
    X_val = X_scaler.transform(X_val)
    X_test = X_scaler.transform(X_test)
    y_encoder.fit(y_train)
    y_train = to_categorical(y_encoder.transform(y_train))
    y_val = to_categorical(y_encoder.transform(y_val))
    y_test = to_categorical(y_encoder.transform(y_test))

    return X_train, X_val, X_test, y_train, y_val, y_test


def prepare_real_estate():
    # read in dataset
    data = pd.read_csv(os.path.join(os.getcwd(), 'RealEstate/Real_estate_valuation_data_set.csv'))

    # create Train, Validation, and Test sets
    data.drop('transaction_date', axis=1, inplace=True)
    X = data.drop('house_price_of_unit_area', axis=1)
    y = data['house_price_of_unit_area']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

    # standardize continuous values to zero mean and unit variance
    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_val = X_scaler.transform(X_val)
    X_test = X_scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


def prepare_ecoli():
    # read in dataset
    data = pd.read_csv(os.path.join(os.getcwd(), 'Ecoli/ecoli.data'), names=['sequence_name', 'mcg', 'gvh', 'lip', 'chg', 
                                                                   'aac', 'alm1', 'alm2', 'class'])
                                                        
    # create Train, Validation, and Test sets
    data.drop('sequence_name', axis=1, inplace=True)
    X = data.drop('class', axis=1)
    y = data['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=5)

    # standardize continuous values to zero mean and unit variance, one-hot encode outputs
    X_scaler = StandardScaler()
    y_encoder = LabelEncoder()
    X_train = X_scaler.fit_transform(X_train)
    X_val = X_scaler.transform(X_val)
    X_test = X_scaler.transform(X_test)
    y_encoder.fit(y_train)
    y_train = to_categorical(y_encoder.transform(y_train))
    y_val = to_categorical(y_encoder.transform(y_val))
    y_test = to_categorical(y_encoder.transform(y_test))

    return X_train, X_val, X_test, y_train, y_val, y_test