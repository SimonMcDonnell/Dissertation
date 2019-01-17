import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from PrepareData import prepare_dataset
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
import os


# plot training history
def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [1000$]')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
           label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
    plt.legend()
    plt.ylim([1, 2])
    plt.show()


def train_linear():
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset()
    model_lin = keras.Sequential([
        keras.layers.Dense(3, input_shape=(10,)),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(1, input_shape=(3,))
    ])
    optimizer = tf.train.GradientDescentOptimizer(0.001)
    model_lin.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    history_lin = model_lin.fit(X_train, y_train, epochs=200, 
        validation_data=[X_val, y_val])
    np.save('linear_weights', model_lin.get_weights(), allow_pickle=True)
    plot_history(history_lin)


def train_sigmoid(batch_norm=True, shift_scale=True):
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset()
    if batch_norm:
        model_sig = keras.Sequential([
            keras.layers.Dense(3, input_shape=(10,)),
            keras.layers.BatchNormalization(center=shift_scale, scale=shift_scale),
            keras.layers.Activation('sigmoid'),
            keras.layers.Dense(1)
        ])
    else:
        model_sig = keras.Sequential([
            keras.layers.Dense(3, input_shape=(10,)),
            keras.layers.Activation('sigmoid'),
            keras.layers.Dense(1)
        ])
    optimizer = tf.train.AdamOptimizer()
    model_sig.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    history_sig = model_sig.fit(X_train, y_train, epochs=300, 
        validation_data=[X_val, y_val])
    np.save('sigmoid_weights', model_sig.get_weights(), allow_pickle=True)
    plot_history(history_sig)


def train_relu():
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset()
    model_relu = keras.Sequential([
        keras.layers.Dense(10, input_shape=(10,)),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Dense(1)
    ])
    optimizer = tf.train.AdamOptimizer(0.001)
    model_relu.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
    history_relu = model_relu.fit(X_train, y_train, epochs=300, 
        validation_data=[X_val, y_val], callbacks=[early_stop])
    np.save('relu_weights', model_relu.get_weights(), allow_pickle=True)
    plot_history(history_relu)

