import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from process_data import prepare_abalone, prepare_concrete, prepare_bank
from sklearn.preprocessing import StandardScaler


# plot training history
def plot_history(history, save=None):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [1000$]')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
           label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
    plt.legend()
    if save != None:
        plt.savefig(save)
    plt.show()


def train_abalone(scale=True, bn=False):
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_abalone()

    # scale outputs
    if scale:
        y_scaler = StandardScaler()
        y_train = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
        y_val = y_scaler.transform(y_val.values.reshape(-1, 1))

    # construct model
    if bn:
        # batch normalization version
        model = keras.Sequential([
            keras.layers.Dense(10, input_shape=(10,)),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),
            keras.layers.Dense(1)
        ])
    else:
        model = keras.Sequential([
            keras.layers.Dense(10, input_shape=(10,), kernel_regularizer=keras.regularizers.l2(0.01)),
            keras.layers.Activation('relu'),
            keras.layers.Dense(1, kernel_regularizer=keras.regularizers.l2(0.01))
        ])
    optimizer = tf.train.AdamOptimizer(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])

    # train
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
    history = model.fit(X_train, y_train, epochs=300, validation_data=[X_val, y_val], callbacks=[early_stop])

    # save
    if scale and bn:
        np.save('weights/abalone/abalone_scale_bn', model.get_weights(), allow_pickle=True)
        plot_history(history, save='graphs/abalone/train/abalone_scale_bn.pdf')
    elif scale:
        np.save('weights/abalone/abalone_scale', model.get_weights(), allow_pickle=True)
        plot_history(history, save='graphs/abalone/train/abalone_scale.pdf')
    elif bn:
        np.save('weights/abalone/abalone_bn', model.get_weights(), allow_pickle=True)
        plot_history(history, save='graphs/abalone/train/abalone_bn.pdf')
    else:
        raise ValueError('Neither scale or batch norm selected')

