# Summary 

Here's a summary of what I've worked on since the last time we Skyped.

## Abalone dataset

I've taken a dataset from UCI where the aim is predict the age of an abalone from physical measurements. I trained a few simple models using keras of various degrees of depth and different activation functions. The details of this are found in *data_exploration.ipynb*. I note that these models, despite being relatively simple, were too complicated for using with seal without expanding on the wrapper. The models are also not ridiculously accurate, I wanted to quickly train something so I could export the weights and try to use them with encrypted data so I didn't spend too much time tuning parameters.

## Seal wrapper

I've expanded on the *seal_wrapper.py* by allowing matrix inputs. Dot products and adding a bias vector to a matrix are now possible.

## Encrypted predictions

Once the seal wrapper was expanded, I attempted to encrypt the test set from the abalone dataset and get some encrypted predictions. This seemed to go well and the details of this can be found in *encrypted_abalone.ipynb*

## Next tasks

* Activation functions - sigmoid, relu
* Networks with hidden layers
* More functionality from seal_wrapper. Functions such as transpose would be nice.

## Questions and points to explore

* It seems that the seal library includes a method for batching inputs so you can work with matrices called PolyCRTBuilder. This seems to depend on setting the encryption parameters to appropriate values and possibly depends on the size of the matrices I intend to use. It seems as well that you can only do element-wise operations on the matrices which means not dot product which isn't ideal...

* The examples provided by PySeal show examples where relinearization is used after doing multiplcations. Is this something that I need to be doing in my current implementation? 

* In addition to the above point, would it be wise to be keep track of the noise budget?

