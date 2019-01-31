import numpy as np
import tensorflow as tf
from tensorflow import keras
from process_data import prepare_abalone, prepare_concrete, prepare_bank
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sys
import time
sys.path.insert(0, 'seal_wrapper/')
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
    ax1.plot(np.arange(0, y.max(), 0.1), np.arange(0, y.max(), 0.1))
    # encrypted
    ax2 = plt.subplot(122)
    ax2.scatter(pred_enc, y, alpha=0.2)
    ax2.set_title(title + ' - enc')
    ax2.set_xlabel('predicted')
    ax2.set_ylabel('true')
    ax2.plot(np.arange(0, y.max(), 0.1), np.arange(0, y.max(), 0.1))
    if save:
        fig.savefig(path)
    plt.show()


def get_weights(data_name, y_train, scale, bn):
    y_scaler = None
    if scale and bn:
        weights = np.load(f'weights/{data_name}/{data_name}_scale_bn.npy')
        w1, b1, gamma, beta, mean, var, w2, b2 = weights
        w1 = normalize_weights(w1, var, gamma)
        b1 = normalize_bias(b1, mean, var, beta, gamma)
        y_scaler = StandardScaler().fit(y_train.values.reshape(-1, 1))
    elif scale:
        weights = np.load(f'weights/{data_name}/{data_name}_scale.npy')
        w1, b1, w2, b2 = weights
        y_scaler = StandardScaler().fit(y_train.values.reshape(-1, 1))
    elif bn:
        weights = np.load(f'weights/{data_name}/{data_name}_bn.npy')
        w1, b1, gamma, beta, mean, var, w2, b2 = weights
        w1 = normalize_weights(w1, var, gamma)
        b1 = normalize_bias(b1, mean, var, beta, gamma)
    else:
        weights = np.load(f'weights/{data_name}/{data_name}.npy')
        w1, b1, w2, b2 = weights
    return [w1, b1, w2, b2, y_scaler]


def run_and_time(X_test, w1, b1, w2, b2, y_scaler=None):
    # clear
    clear_start = time.time()
    l1_clear = X_test.dot(w1) + b1
    l1_relu_clear = relu(l1_clear)
    pred_clear = np.dot(l1_relu_clear, w2) + b2
    clear_end = time.time()
    if y_scaler != None:
        pred_clear = y_scaler.inverse_transform(pred_clear.flatten())
    
    # encrypted
    X_test_enc = EA(X_test, True)
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
    if y_scaler != None:
        pred_enc = y_scaler.inverse_transform(pred_enc.flatten())

    return [pred_clear, pred_enc, clear_end, clear_start, enc_end, enc_start]


def save_results(data_name, pred_clear, pred_enc, y_test, clear_end, clear_start, enc_end, enc_start, scale, bn):   
    # report predictions
    print('Clear: MSSE = {};\tRuntime = {}s'.format(msse(pred_clear.flatten(), y_test), (clear_end-clear_start)))
    print('Encrypted: MSSE = {};\tRuntime = {}s'.format(msse(pred_enc.flatten(), y_test), (enc_end-enc_start)))
    if scale and bn:
        plot_predictions(pred_clear.flatten(), pred_enc.flatten(), y_test, 
            f'{data_name}', f'graphs/{data_name}/test/{data_name}_scale_bn.pdf')
        file = open(f'graphs/{data_name}/test/{data_name}_scale_bn.txt', 'w')
        file.write('Clear: MSSE = {};\tRuntime = {}s\n'.format(msse(pred_clear.flatten(), y_test), (clear_end-clear_start)))
        file.write('Encrypted: MSSE = {};\tRuntime = {}s'.format(msse(pred_enc.flatten(), y_test), (enc_end-enc_start)))
        file.close()
    elif scale:
        plot_predictions(pred_clear.flatten(), pred_enc.flatten(), y_test, 
            f'{data_name}', f'graphs/{data_name}/test/{data_name}_scale.pdf')
        file = open(f'graphs/{data_name}/test/{data_name}_scale.txt', 'w')
        file.write('Clear: MSSE = {};\tRuntime = {}s\n'.format(msse(pred_clear.flatten(), y_test), (clear_end-clear_start)))
        file.write('Encrypted: MSSE = {};\tRuntime = {}s'.format(msse(pred_enc.flatten(), y_test), (enc_end-enc_start)))
        file.close()
    elif bn:
        plot_predictions(pred_clear.flatten(), pred_enc.flatten(), y_test, 
            f'{data_name}', f'graphs/{data_name}/test/{data_name}_bn.pdf')
        file = open(f'graphs/{data_name}/test/{data_name}_bn.txt', 'w')
        file.write('Clear: MSSE = {};\tRuntime = {}s\n'.format(msse(pred_clear.flatten(), y_test), (clear_end-clear_start)))
        file.write('Encrypted: MSSE = {};\tRuntime = {}s'.format(msse(pred_enc.flatten(), y_test), (enc_end-enc_start)))
        file.close()
    else:
        plot_predictions(pred_clear.flatten(), pred_enc.flatten(), y_test, 
            f'{data_name}', f'graphs/{data_name}/test/{data_name}.pdf')
        file = open(f'graphs/{data_name}/test/{data_name}.txt', 'w')
        file.write('Clear: MSSE = {};\tRuntime = {}s\n'.format(msse(pred_clear.flatten(), y_test), (clear_end-clear_start)))
        file.write('Encrypted: MSSE = {};\tRuntime = {}s'.format(msse(pred_enc.flatten(), y_test), (enc_end-enc_start)))
        file.close()


def eval_abalone(scale=True, bn=False):
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_abalone()
    w1, b1, w2, b2, y_scaler = get_weights('abalone', y_train, scale, bn)
    results = run_and_time(X_test, w1, b1, w2, b2, y_scaler)
    pred_clear, pred_enc, clear_end, clear_start, enc_end, enc_start = results
    save_results('abalone', pred_clear, pred_enc, y_test, clear_end, clear_start, enc_end, enc_start, scale, bn)


def eval_concrete(scale=True, bn=False):
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_concrete()
    w1, b1, w2, b2, y_scaler = get_weights('concrete', y_train, scale, bn)
    results = run_and_time(X_test, w1, b1, w2, b2, y_scaler)
    pred_clear, pred_enc, clear_end, clear_start, enc_end, enc_start = results
    save_results('concrete', pred_clear, pred_enc, y_test, clear_end, clear_start, enc_end, enc_start, scale, bn)


# NOT COMPLETE
def eval_bank(scale=False, bn=False):
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_bank()
    w1, b1, w2, b2, y_scaler = get_weights('bank', y_train, scale, bn)
    results = run_and_time(X_test, w1, b1, w2, b2, y_scaler)
    pred_clear, pred_enc, clear_end, clear_start, enc_end, enc_start = results
    pred_clear = sigmoid(pred_clear)
    pred_enc = sigmoid(pred_enc)
    save_results('bank', pred_clear, pred_enc, y_test, clear_end, clear_start, enc_end, enc_start, scale, bn)
    