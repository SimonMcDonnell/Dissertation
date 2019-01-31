import numpy as np
import tensorflow as tf
from tensorflow import keras
from process_data import prepare_abalone, prepare_concrete, prepare_bank
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sys
import time
sys.path.insert(0, '../seal_wrapper/')
from seal_wrapper import EA


# mean sum squared error
def msse(pred, y):
    return np.sum((y - pred)**2) / y.shape[0]


def normalize_weights(w, var, gamma):
    return w * gamma / np.sqrt(var + 0.0001)


def normalize_bias(b, mean, var, beta, gamma):
    return (b - mean) * (gamma / np.sqrt(var + 0.0001)) + beta


def sigmoid(z):
    return 1/(1+np.exp(-z))


def relu(z):
    return np.maximum(0, z)


def plot_predictions(pred_clear, pred_enc, y, title, path, save=True):
    fig = plt.figure(figsize=(7, 7))
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
    if save:
        fig.savefig(title + '.pdf')
    plt.show()


def eval_abalone(scale=True, bn=False):
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_abalone()
    # load weights
    if scale and bn:
        weights = np.load('weights/abalone/abalone_scale_bn.npy')
        w1, b1, gamma, beta, mean, var, w2, b2 = weights
        w1 = normalize_weights(w1, var, gamma)
        b1 = normalize_bias(b1, mean, var, beta, gamma)
        y_scaler = StandardScaler().fit(y_train.values.reshape(-1, 1))
    elif scale:
        weights = np.load('weights/abalone/abalone_scale.npy')
        w1, b1, w2, b2 = weights
        y_scaler = StandardScaler().fit(y_train.values.reshape(-1, 1))
    elif bn:
        weights = np.load('weights/abalone/abalone_bn.npy')
        w1, b1, gamma, beta, mean, var, w2, b2 = weights
        w1 = normalize_weights(w1, var, gamma)
        b1 = normalize_bias(b1, mean, var, beta, gamma)
    else:
        raise ValueError('No weights for no scale or batch norm')
    
    # clear
    clear_start = time.time()
    l1_clear = X_test.values.dot(w1) + b1
    l1_relu_clear = relu(l1_clear)
    pred_clear = np.dot(l1_relu_clear, w2) + b2
    clear_end = time.time()
    if scale:
        pred_clear = y_scaler.inverse_transform(pred_clear.flatten())
    
    # encrypted
    X_test_enc = EA(X_test.values, True)
    w1_enc = EA(w1)
    b1_enc = EA(b1.reshape(1, -1))
    w2_enc = EA(w2)
    b2_enc = EA(b2.reshape(1, -1))
    enc_start = time.time()
    l1_enc = X_test_enc.dot(w1_enc) + b1_enc
    l1_relu_enc = l1_enc.relu()
    pred_enc = l1_relu_enc.dot(w2_enc) + b2_enc
    enc_end = time.time()
    pred_enc = pred_enc.values()
    if scale:
        pred_enc = y_scaler.inverse_transform(pred_enc.flatten())
    
    # report predictions
    print('Clear: MSSE = {};\tRuntime = {}s'.format(msse(pred_clear.flatten(), y_test)), (clear_end-clear_start))
    print('Encrypted: MSSE = {};\tRuntime = {}s'.format(msse(pred_enc.flatten(), y_test)), (enc_end-enc_start))
    if scale and bn:
        plot_predictions(pred_clear.flatten(), pred_enc.flatten(), y_test, 
            'Abalone', 'graphs/abalone/test/abalone_scale_bn.pdf')
    elif scale:
        plot_predictions(pred_clear.flatten(), pred_enc.flatten(), y_test, 
            'Abalone', 'graphs/abalone/test/abalone_scale.pdf')
    elif bn:
        plot_predictions(pred_clear.flatten(), pred_enc.flatten(), y_test, 
            'Abalone', 'graphs/abalone/test/abalone_bn.pdf')
    else:
        raise ValueError('Neither scale or batch norm selected')