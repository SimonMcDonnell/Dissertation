import numpy as np
np.random.seed(42)
import random as rn
rn.seed(12345)
import tensorflow as tf
tf.set_random_seed(82)
from tensorflow import keras
from sklearn.metrics import accuracy_score, confusion_matrix
from process_data import prepare_abalone, prepare_concrete, prepare_bank, prepare_iris, prepare_real_estate, prepare_ecoli
import matplotlib.pyplot as plt
import seaborn as sns
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


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1).reshape(-1, 1)


def sigmoid(z):
    return 1/(1+np.exp(-z))


def relu(z):
    return np.maximum(0, z)


def plot_predictions(pred_clear, pred_enc, y, title, path, save=True):
    fig = plt.figure(figsize=(9, 5))
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


def plot_final_layer(pred_clear, pred_enc, title, path, save=True):
    fig = plt.figure(figsize=(7, 7))
    # clear
    ax1 = plt.subplot(111)
    ax1.scatter(pred_enc, pred_clear, alpha=0.2)
    ax1.set_title(title)
    ax1.set_xlabel('Encrypted')
    ax1.set_ylabel('Clear')
    ax1.plot(np.arange(pred_clear.min(), pred_clear.max(), 0.1), np.arange(pred_clear.min(), pred_clear.max(), 0.1))
    if save:
        fig.savefig(path)
    plt.show()


def plot_first_layer(l1, scale, bn, path, save=True):
    plt.figure(figsize=(7, 7))
    plt.hist(l1.flatten(), bins=50)
    if save:
        if scale and bn:
            plt.savefig(path + '_scale_bn.pdf')
        elif scale:
            plt.savefig(path + '_scale.pdf')
        elif bn:
            plt.savefig(path + '_bn.pdf')
        else:
            plt.savefig(path + '.pdf')


def plot_confusion(pred_clear, pred_enc, y, classes, title, path, save=True):
    mat_clear = confusion_matrix(pred_clear, y)
    cm_clear = mat_clear / np.sum(mat_clear, axis=1).reshape(-1, 1)
    mat_enc = confusion_matrix(pred_enc, y)
    cm_enc = mat_enc / np.sum(mat_enc, axis=1).reshape(-1, 1)
    fig = plt.figure(figsize=(9, 5))
    # clear
    ax1 = plt.subplot(121)
    if classes is not None:
        sns.heatmap(cm_clear, xticklabels=classes, yticklabels=classes, vmin=0., vmax=1., annot=True, ax=ax1)
    else:
        sns.heatmap(cm_clear, vmin=0., vmax=1., ax=ax1)
    ax1.set_title(title + ' - clear')
    # encrypted
    ax2 = plt.subplot(122)
    if classes is not None:
        sns.heatmap(cm_enc, xticklabels=classes, yticklabels=classes, vmin=0., vmax=1., annot=True, ax=ax2)
    else:
        sns.heatmap(cm_enc, vmin=0., vmax=1., ax=ax2)
    ax2.set_title(title + ' - enc')
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

    return [pred_clear, pred_enc, clear_end, clear_start, enc_end, enc_start, l1_enc.values()]


def save_reg_results(data_name, pred_clear, pred_enc, y_test, clear_end, clear_start, enc_end, enc_start, scale, bn):   
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


def save_class_results(data_name, pred_clear, pred_enc, y_test, clear_end, clear_start, enc_end, enc_start, classes, scale, bn):   
    # report predictions
    print('Clear: Accuracy = {};\tRuntime = {}s'.format(accuracy_score(pred_clear.round(), y_test), (clear_end-clear_start)))
    print('Encrypted: Accuracy = {};\tRuntime = {}s'.format(accuracy_score(pred_enc.round(), y_test), (enc_end-enc_start)))
    if scale and bn:
        plot_confusion(pred_clear, pred_enc, y_test, classes,
            f'{data_name}', f'graphs/{data_name}/test/{data_name}_cm_scale_bn.pdf')
        file = open(f'graphs/{data_name}/test/{data_name}_scale_bn.txt', 'w')
        file.write('Clear: Accuracy = {};\tRuntime = {}s\n'.format(accuracy_score(pred_clear, y_test), (clear_end-clear_start)))
        file.write('Encrypted: Accuracy = {};\tRuntime = {}s'.format(accuracy_score(pred_enc, y_test), (enc_end-enc_start)))
        file.close()
    elif scale:
        plot_confusion(pred_clear, pred_enc, y_test, classes,
            f'{data_name}', f'graphs/{data_name}/test/{data_name}_cm_scale.pdf')
        file = open(f'graphs/{data_name}/test/{data_name}_scale.txt', 'w')
        file.write('Clear: Accuracy = {};\tRuntime = {}s\n'.format(accuracy_score(pred_clear, y_test), (clear_end-clear_start)))
        file.write('Encrypted: Accuracy = {};\tRuntime = {}s'.format(accuracy_score(pred_enc, y_test), (enc_end-enc_start)))
        file.close()
    elif bn:
        plot_confusion(pred_clear, pred_enc, y_test, classes,
            f'{data_name}', f'graphs/{data_name}/test/{data_name}_cm_bn.pdf')
        file = open(f'graphs/{data_name}/test/{data_name}_bn.txt', 'w')
        file.write('Clear: Accuracy = {};\tRuntime = {}s\n'.format(accuracy_score(pred_clear, y_test), (clear_end-clear_start)))
        file.write('Encrypted: Accuracy = {};\tRuntime = {}s'.format(accuracy_score(pred_enc, y_test), (enc_end-enc_start)))
        file.close()
    else:
        plot_confusion(pred_clear, pred_enc, y_test, classes,
            f'{data_name}', f'graphs/{data_name}/test/{data_name}_cm.pdf')
        file = open(f'graphs/{data_name}/test/{data_name}.txt', 'w')
        file.write('Clear: Accuracy = {};\tRuntime = {}s\n'.format(accuracy_score(pred_clear, y_test), (clear_end-clear_start)))
        file.write('Encrypted: Accuracy = {};\tRuntime = {}s'.format(accuracy_score(pred_enc, y_test), (enc_end-enc_start)))
        file.close()


def eval_abalone(scale=True, bn=False):
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_abalone()
    w1, b1, w2, b2, y_scaler = get_weights('abalone', y_train, scale, bn)
    results = run_and_time(X_test, w1, b1, w2, b2, y_scaler)
    pred_clear, pred_enc, clear_end, clear_start, enc_end, enc_start, l1 = results
    plot_first_layer(l1, scale, bn, 'graphs/abalone/test/first_layer')
    save_reg_results('abalone', pred_clear, pred_enc, y_test, clear_end, clear_start, enc_end, enc_start, scale, bn)


def eval_concrete(scale=True, bn=False):
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_concrete()
    w1, b1, w2, b2, y_scaler = get_weights('concrete', y_train, scale, bn)
    results = run_and_time(X_test, w1, b1, w2, b2, y_scaler)
    pred_clear, pred_enc, clear_end, clear_start, enc_end, enc_start, l1 = results
    plot_first_layer(l1, scale, bn, 'graphs/concrete/test/first_layer')
    save_reg_results('concrete', pred_clear, pred_enc, y_test, clear_end, clear_start, enc_end, enc_start, scale, bn)


def eval_bank(scale=False, bn=False):
    if scale:
        raise ValueError('Cannot scale outputs for classification tasks')
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_bank()
    w1, b1, w2, b2, y_scaler = get_weights('bank', y_train, scale, bn)
    results = run_and_time(X_test, w1, b1, w2, b2, y_scaler)
    pred_clear, pred_enc, clear_end, clear_start, enc_end, enc_start, l1 = results
    plot_first_layer(l1, scale, bn, 'graphs/bank/test/first_layer')
    pred_clear_ = np.round(sigmoid(pred_clear))
    pred_enc_ = np.round(sigmoid(pred_enc))
    classes = ['Real', 'Fake']
    if bn:
        plot_final_layer(pred_clear, pred_enc, 'Final layer predictions', 'graphs/bank/test/bank_final_act_bn.pdf')
    else:
        plot_final_layer(pred_clear, pred_enc, 'Final layer predictions', 'graphs/bank/test/bank_final_act.pdf')
    save_class_results('bank', pred_clear_, pred_enc_, y_test, clear_end, clear_start, enc_end, enc_start, classes, scale, bn)


def eval_iris(scale=False, bn=False):
    if scale:
        raise ValueError('Cannot scale outputs for classification tasks')
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_iris()
    w1, b1, w2, b2, y_scaler = get_weights('iris', y_train, scale, bn)
    results = run_and_time(X_test, w1, b1, w2, b2, y_scaler)
    pred_clear, pred_enc, clear_end, clear_start, enc_end, enc_start, l1 = results
    plot_first_layer(l1, scale, bn, 'graphs/iris/test/first_layer')
    pred_clear_ = np.argmax(softmax(pred_clear), axis=1)
    pred_enc_ = np.argmax(softmax(pred_enc), axis=1)
    y_test = np.argmax(y_test, axis=1)
    classes = ['setosa', 'versicolor', 'virginica']
    if bn:
        plot_final_layer(pred_clear.max(axis=1), pred_enc.max(axis=1), 'Final layer predictions', 'graphs/iris/test/iris_final_act_bn.pdf')
    else:
        plot_final_layer(pred_clear.max(axis=1), pred_enc.max(axis=1), 'Final layer predictions', 'graphs/iris/test/iris_final_act.pdf')
    save_class_results('iris', pred_clear_, pred_enc_, y_test, clear_end, clear_start, enc_end, enc_start, classes, scale, bn)


def eval_real_estate(scale=True, bn=False):
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_real_estate()
    w1, b1, w2, b2, y_scaler = get_weights('real_estate', y_train, scale, bn)
    results = run_and_time(X_test, w1, b1, w2, b2, y_scaler)
    pred_clear, pred_enc, clear_end, clear_start, enc_end, enc_start, l1 = results
    plot_first_layer(l1, scale, bn, 'graphs/real_estate/test/first_layer')
    save_reg_results('real_estate', pred_clear, pred_enc, y_test, clear_end, clear_start, enc_end, enc_start, scale, bn)


def eval_ecoli(scale=False, bn=False):
    if scale:
        raise ValueError('Cannot scale outputs for classification tasks')
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_ecoli()
    w1, b1, w2, b2, y_scaler = get_weights('ecoli', y_train, scale, bn)
    results = run_and_time(X_test, w1, b1, w2, b2, y_scaler)
    pred_clear, pred_enc, clear_end, clear_start, enc_end, enc_start, l1 = results
    plot_first_layer(l1, scale, bn, 'graphs/ecoli/test/first_layer')
    pred_clear_ = np.argmax(softmax(pred_clear), axis=1)
    pred_enc_ = np.argmax(softmax(pred_enc), axis=1)
    y_test = np.argmax(y_test, axis=1)
    classes = ['cp', 'im', 'pp', 'imU', 'om', 'omL', 'imL', 'imS']
    if bn:
        plot_final_layer(pred_clear.max(axis=1), pred_enc.max(axis=1), 'Final layer predictions', 'graphs/ecoli/test/ecoli_final_act_bn.pdf')
    else:
        plot_final_layer(pred_clear.max(axis=1), pred_enc.max(axis=1), 'Final layer predictions', 'graphs/ecoli/test/ecoli_final_act.pdf')
    save_class_results('ecoli', pred_clear_, pred_enc_, y_test, clear_end, clear_start, enc_end, enc_start, classes, scale, bn)