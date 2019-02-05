import numpy as np
np.random.seed(42)
import random as rn
rn.seed(12345)
import tensorflow as tf
tf.set_random_seed(82)
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow import set_random_seed
from process_data import prepare_abalone, prepare_concrete, prepare_bank, prepare_iris, prepare_real_estate
from sklearn.preprocessing import StandardScaler

set_random_seed(1)


# plot training history
def plot_history(history, type='reg', save=None):
    plt.figure()
    plt.xlabel('Epoch')
    if type == 'reg':
        plt.ylabel('Mean Abs Error [1000$]')
        plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
            label='Train Loss')
        plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
            label = 'Val loss')
    else:
        plt.ylabel('Mean Abs Error [1000$]')
        plt.plot(history.epoch, np.array(history.history['loss']),
            label='Train Loss')
        plt.plot(history.epoch, np.array(history.history['val_loss']),
            label = 'Val loss')
    plt.legend()
    if save != None:
        plt.savefig(save)
    plt.show()


def save_weights_graphs(data_name, scale, bn, model, history, type):
    if scale and bn:
        np.save(f'weights/{data_name}/{data_name}_scale_bn', model.get_weights(), allow_pickle=True)
        plot_history(history, type, save=f'graphs/{data_name}/train/{data_name}_scale_bn.pdf')
    elif scale:
        np.save(f'weights/{data_name}/{data_name}_scale', model.get_weights(), allow_pickle=True)
        plot_history(history, type, save=f'graphs/{data_name}/train/{data_name}_scale.pdf')
    elif bn:
        np.save(f'weights/{data_name}/{data_name}_bn', model.get_weights(), allow_pickle=True)
        plot_history(history, type, save=f'graphs/{data_name}/train/{data_name}_bn.pdf')
    else:
        np.save(f'weights/{data_name}/{data_name}', model.get_weights(), allow_pickle=True)
        plot_history(history, type, save=f'graphs/{data_name}/train/{data_name}.pdf')


def train_abalone(scale=True, bn=False):
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_abalone()
    if scale:
        y_scaler = StandardScaler()
        y_train = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
        y_val = y_scaler.transform(y_val.values.reshape(-1, 1))

    # construct model
    if bn:
        # batch normalization version
        model = keras.Sequential([
            keras.layers.Dense(3, input_shape=(X_train.shape[1],)),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            keras.layers.Dense(1)
        ])
    else:
        model = keras.Sequential([
            keras.layers.Dense(3, input_shape=(X_train.shape[1],), kernel_regularizer=keras.regularizers.l2(0.01)),
            keras.layers.Activation('relu'),
            keras.layers.Dense(1, kernel_regularizer=keras.regularizers.l2(0.01))
        ])
    optimizer = tf.train.AdamOptimizer(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])

    # train
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
    history = model.fit(X_train, y_train, epochs=1000, validation_data=[X_val, y_val], callbacks=[early_stop])
    # save
    save_weights_graphs('abalone', scale, bn, model, history, 'reg')


def train_concrete(scale=True, bn=False):
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_concrete()
    if scale:
        y_scaler = StandardScaler()
        y_train = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
        y_val = y_scaler.transform(y_val.values.reshape(-1, 1))

    # construct model
    if bn:
        # batch normalization version
        model = keras.Sequential([
            keras.layers.Dense(8, input_shape=(X_train.shape[1],)),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            keras.layers.Dense(1)
        ])
    else:
        model = keras.Sequential([
            keras.layers.Dense(8, input_shape=(X_train.shape[1],), kernel_regularizer=keras.regularizers.l2(0.01)),
            keras.layers.Activation('relu'),
            keras.layers.Dense(1, kernel_regularizer=keras.regularizers.l2(0.01))
        ])
    optimizer = tf.train.AdamOptimizer(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])

    # train
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
    history = model.fit(X_train, y_train, epochs=1000, validation_data=[X_val, y_val], callbacks=[early_stop])
    # save
    save_weights_graphs('concrete', scale, bn, model, history, 'reg')


def train_bank(scale=False, bn=False):
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_bank()
    if scale:
        raise ValueError('Cannot scale outputs for classification tasks')

    # construct model
    if bn:
        # batch normalization version
        model = keras.Sequential([
            keras.layers.Dense(4, input_shape=(X_train.shape[1],)),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            keras.layers.Dense(1),
            keras.layers.Activation('sigmoid')
        ])
    else:
        model = keras.Sequential([
            keras.layers.Dense(4, input_shape=(X_train.shape[1],), kernel_regularizer=keras.regularizers.l2(0.01)),
            keras.layers.Activation('relu'),
            keras.layers.Dense(1, kernel_regularizer=keras.regularizers.l2(0.01)),
            keras.layers.Activation('sigmoid')
        ])
    optimizer = tf.train.AdamOptimizer(0.001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # train
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
    history = model.fit(X_train, y_train, epochs=1000, validation_data=[X_val, y_val], callbacks=[early_stop])
    # save
    save_weights_graphs('bank', scale, bn, model, history, 'class')


def train_iris(scale=False, bn=False):
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_iris()
    if scale:
        raise ValueError('Cannot scale outputs for classification tasks')

    # construct model
    if bn:
        model = keras.Sequential([
            keras.layers.Dense(4, input_shape=(4,)),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            keras.layers.Dense(3),
            keras.layers.Activation('softmax')
        ])
    else:
        model = keras.Sequential([
            keras.layers.Dense(4, input_shape=(4,), kernel_regularizer=keras.regularizers.l2(0.01)),
            keras.layers.Activation('relu'),
            keras.layers.Dense(3, kernel_regularizer=keras.regularizers.l2(0.01)),
            keras.layers.Activation('softmax')
        ])
    optimizer = tf.train.AdamOptimizer(0.005)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # train
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
    history = model.fit(X_train, y_train, epochs=1000, validation_data=[X_val, y_val], callbacks=[early_stop])
    # save
    save_weights_graphs('iris', scale, bn, model, history, 'class')


def train_real_estate(scale=True, bn=False):
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_real_estate()
    if scale:
        y_scaler = StandardScaler()
        y_train = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
        y_val = y_scaler.transform(y_val.values.reshape(-1, 1))

    # construct model
    if bn:
        # batch normalization version
        model = keras.Sequential([
            keras.layers.Dense(5, input_shape=(X_train.shape[1],)),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            keras.layers.Dense(1)
        ])
    else:
        model = keras.Sequential([
            keras.layers.Dense(5, input_shape=(X_train.shape[1],), kernel_regularizer=keras.regularizers.l2(0.01)),
            keras.layers.Activation('relu'),
            keras.layers.Dense(1, kernel_regularizer=keras.regularizers.l2(0.01))
        ])
    optimizer = tf.train.AdamOptimizer(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])

    # train
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
    history = model.fit(X_train, y_train, epochs=1000, validation_data=[X_val, y_val], callbacks=[early_stop])
    # save
    save_weights_graphs('real_estate', scale, bn, model, history, 'reg')