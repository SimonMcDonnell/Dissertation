# Dissertation log

## September

### 21/09/18

* Installed Docker and created image for PySeal
    * Docker is run like this: `docker run -it -p 8888:8888 seal-save jupyter notebook --ip 0.0.0.0 --no-browser --allow-root`
    * Then type `localhost:8888/tree` into browser
    * Insert token from terminal
* Worked through `demo.ipynb`

### 23/09/18

* Researched more papers to read discussing homomorphic encryption standards, 
different encryption schemes, and using homomorphic encryption in neural networks

### 26/09/18

* Read paper - *CryptoNets: Applying Neural Networks to Encrypted Data with High Throughput and Accuracy*
* Sent email to Yongxin describing what I've looked at and worked on so far

### 30/09/18

* Trained neural networks on UCI abalone dataset in data_exploraion.ipynb
* Saved weights for multiple models trained on this data
* Installed VirtualBox Ubuntu image with a plan to install PySeal natively next time if I can

## October

### 01/10/18

* Used sigmoid as activation function for abalone dataset and saved weights

### 05/10/18

* Installed PySeal natively on Ubuntu, no need for Docker anymore

### 14/10/18

* Updated seal_wrapper to include the following functionality:
    * Dot product for matrices of any size
    * Decrypt matrices of any size
* Predicted results using encrypted Abalone data and pre-trained weights from linear model
* Compared predicted results to true labels, found some discrepancy, however, looks to just be a bias issue