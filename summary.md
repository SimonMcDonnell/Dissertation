# Summary

## Batch Normalization

I've implemented batch normalization using the method you linked previously by absorbing the normalization layer into the weights and biases. I have trained and tested two new networks, on the same Abalone dataset, one using a sigmoid activation approximation and one with a relu approximation (more on that below).

## Activation functions

I have attemped to approximate the sigmoid and relu activation function, however, I am having difficulties. I am approximating the sigmoid function using the first two terms of the Maclaurin series, which is not a great approximation. Unfortunately when I tried using more terms, the errors would become very significant. I believe this is due to rounding errors and figuring out relinearization may be a solution to this? I approximated relu using the simple approximation used in the paper CryptoNets, x^2. This again did not perform that well.

## Files

* __seal_wrapper.py__ - updated to include activation function approximations. These do not function well and there is a lot of redundant code in an attempt to fix the issues.
* __data_exploration.ipynb__ - python notebook containing the trained models. This is where I create the different models and save their weights.
* __encrypted_abalone.ipynb__ - python notebook where encrypted experiments are conducted. I load the saved weights and get encrypted results. There are experiments with batch normalization and activations in this notebook.