import numpy as np
import tensorflow as tf
from tensorflow import keras
from PrepareData import prepare_dataset
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../seal_wrapper/')
from seal_wrapper import EA

# mean sum squared error
def msse(pred, y):
    return np.sum((y - pred)**2) / y.shape[0]


def plot_predictions(pred, y, title):
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111)
    ax.scatter(pred, y, alpha=0.2)
    ax.title(title)
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')
    ax.plot(np.arange(0, 30, 0.1), np.arange(0, 30, 0.1))


def eval_linear():
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset()
    w, b = np.load('linear_weights.npy')
    # clear
    pred_clear = np.dot(X_test, w) + b
    # encrypted
    X_test_enc = EA(X_test.values, True)
    w_enc = EA(w)
    b_enc = EA(b.reshape(1, -1))
    pred_enc = (X_test_enc.dot(w_enc) + b_enc).values()
    # report predictions
    print('MSSE clear: {}'.format(msse(pred_clear.flatten(), y_test)))
    print('MSSE enc: {}'.format(msse(pred_enc.flatten(), y_test)))
    plot_predictions(pred_clear.flatten(), y_test, 'Linear - clear')
    plot_predictions(pred_enc.flatten(), y_test, 'Linear - enc')
