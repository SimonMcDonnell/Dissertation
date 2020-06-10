# Privacy Preserving Deep Learning
This project will analyse using Deep Learning in conjunction with Homomorphic Encryption. I will be using Microsoft's SEAL library for encrypting the data before passing it through different Neural Network architectures with various different activation functions. The performance of using SEAL will then be assessed.   

Models are trained in train_models.py
Models are evaluated in eval_models.py
Data sets are prepared in process_data.py
Python wrapper for encrypted operations in seal_wrapper/seal_wrapper.py
Approximation of activation functions in relu_approx.ipynb
