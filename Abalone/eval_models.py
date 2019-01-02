import numpy as np
import tensorflow as tf
from tensorflow import keras
from PrepareData import prepare_dataset
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../seal_wrapper/')
from seal_wrapper import EA

X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset()
X_test_enc = EA(X_test.values, True)

# mean sum squared error
def msse(pred, y):
    return np.sum((y - pred)**2) / y.shape[0]


def normalize_weights(w, var, scale):
    return w * scale / np.sqrt(var + 0.001)


def normalize_bias(b, mean, var, shift, scale):
    return (b - mean) * (scale / np.sqrt(var + 0.001)) + shift


def sigmoid(z):
    return 1/(1+np.exp(-z))


def plot_predictions(pred_clear, pred_enc, y, title):
    fig = plt.figure(figsize=(6, 6))
    # clear
    ax1 = plt.subplot(121)
    ax1.scatter(pred_clear, y, alpha=0.2)
    ax1.set_title(title + ' - clear')
    ax1.set_xlabel('predicted')
    ax1.set_ylabel('true')
    ax1.plot(np.arange(0, 30, 0.1), np.arange(0, 30, 0.1))
    # encrypted
    ax2 = plt.subplot(122)
    ax2.scatter(pred_enc, y, alpha=0.2)
    ax2.set_title(title + ' - enc')
    ax2.set_xlabel('predicted')
    ax2.set_ylabel('true')
    ax2.plot(np.arange(0, 30, 0.1), np.arange(0, 30, 0.1))
    plt.show()


def eval_linear():
    weights = np.load('linear_weights.npy')
    w1, b1, scale, shift, mean, std, w2, b2 = weights
    w1 = normalize_weights(w1, std**2, scale)
    b1 = normalize_bias(b1, mean, std**2, shift, scale).reshape(1, -1)
    # clear
    l1_clear = X_test.values.dot(w1) + b1
    pred_clear = np.dot(l1_clear, w2) + b2
    # encrypted
    w1_enc = EA(w1)
    b1_enc = EA(b1.reshape(1, -1))
    w2_enc = EA(w2)
    b2_enc = EA(b2.reshape(1, -1))
    l1_enc = X_test_enc.dot(w1_enc) + b1_enc
    pred_enc = (l1_enc.dot(w2_enc) + b2_enc).values()
    # report predictions
    print('MSSE clear: {}'.format(msse(pred_clear.flatten(), y_test)))
    print('MSSE enc: {}'.format(msse(pred_enc.flatten(), y_test)))
    plot_predictions(pred_clear.flatten(), pred_enc.flatten(), y_test, 'Linear')


def eval_sigmoid():
    weights = np.load('sigmoid_weights.npy')
    w1, b1, scale, shift, mean, std, w2, b2 = weights
    w1 = normalize_weights(w1, std**2, scale)
    b1 = normalize_bias(b1, mean, std**2, shift, scale).reshape(1, -1)
    # clear
    l1_clear = X_test.values.dot(w1) + b1
    l1_sig_clear = sigmoid(l1_clear)
    pred_clear = np.dot(l1_sig_clear, w2) + b2
    # encrypted
    w1_enc = EA(w1)
    b1_enc = EA(b1.reshape(1, -1))
    w2_enc = EA(w2)
    b2_enc = EA(b2.reshape(1, -1))
    l1_enc = X_test_enc.dot(w1_enc) + b1_enc
    l1_sig_enc = l1_enc.sigmoid()
    pred_enc = (l1_sig_enc.dot(w2_enc) + b2_enc).values()
    # report predictions
    print('MSSE clear: {}'.format(msse(pred_clear.flatten(), y_test)))
    print('MSSE enc: {}'.format(msse(pred_enc.flatten(), y_test)))
    plot_predictions(pred_clear.flatten(), pred_enc.flatten(), y_test, 'Sigmoid')


def eval_relu():
    weights = np.load('relu_weights.npy')
    w1, b1, scale, shift, mean, std, w2, b2 = weights
    w1 = normalize_weights(w1, std**2, scale)
    b1 = normalize_bias(b1, mean, std**2, shift, scale).reshape(1, -1)
    # clear
    l1_clear = X_test.values.dot(w1) + b1
    l1_relu_clear = relu(l1_clear)
    pred_clear = np.dot(l1_relu_clear, w2) + b2
    # encrypted
    w1_enc = EA(w1)
    b1_enc = EA(b1.reshape(1, -1))
    w2_enc = EA(w2)
    b2_enc = EA(b2.reshape(1, -1))
    l1_enc = X_test_enc.dot(w1_enc) + b1_enc
    l1_relu_enc = l1_enc.relu()
    pred_enc = (l1_relu_enc.dot(w2_enc) + b2_enc).values()
    # report predictions
    print('MSSE clear: {}'.format(msse(pred_clear.flatten(), y_test)))
    print('MSSE enc: {}'.format(msse(pred_enc.flatten(), y_test)))
    plot_predictions(pred_clear.flatten(), pred_enc.flatten(), y_test, 'Relu')