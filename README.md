# DeepLearning

## Table of Content

- [MNIST Dense Neural Network](#MNIST-DenseNeuralNetwork)
- [MNIST SVM](#MNIST-SVM)

# MNIST-DenseNeuralNetwork

import numpy as np: NumPy is a library for numerical operations in Python. It provides support for large, multi-dimensional arrays and matrices, along with mathematical functions to operate on these elements.

import matplotlib.pyplot as plt: Matplotlib is a plotting library for Python. The alias plt is a commonly used convention.

import keras: Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. It provides a convenient way to define and train deep learning models.

from keras.datasets import mnist: This import is specifically for the MNIST dataset, which is a commonly used dataset for training and testing image classification models.

from keras.models import Sequential, model_from_json: The Sequential class is used to create a linear stack of layers for building a neural network. model_from_json is used to load a model from a JSON file.

from keras.layers import Dense: The Dense layer is used to create a fully connected neural network layer.

from keras.optimizers import RMSprop: RMSprop is an optimization algorithm commonly used for training neural networks. It adapts the learning rates of each parameter during training.

import pylab as plt: This is redundant since you have already imported matplotlib.pyplot as plt earlier. You can choose to keep one of them.

batch_size: The number of training examples utilized in one iteration. This is a hyperparameter that can be tuned based on your system's memory capacity.

num_classes: The number of classes in the classification task. For MNIST, it's 10, as there are 10 digits (0 through 9).

epochs: The number of times the model will iterate over the entire training dataset. Each epoch consists of one forward pass and one backward pass of all the training examples.

when you reshape x_train with the dimensions (60000, 784), you are flattening each 28x28 image into a 1D array of size 784. This is a common practice when working with fully connected (dense) layers in a neural network

One-hot encoding is a binary matrix representation of the labels. In this representation, each label is converted into a binary vector of length equal to the number of classes (num_classes). The vector is all zeros except for the index that corresponds to the class, which is marked with a 1. For example:

If num_classes is 10 and the original label is 3, it will be one-hot encoded as [0, 0, 0, 1, 0, 0, 0, 0, 0, 0].
If the original label is 7, it will be one-hot encoded as [0, 0, 0, 0, 0, 0, 0, 1, 0, 0].
This one-hot encoding is often used in the output layer of a neural network for classification tasks. It helps the network understand the categorical nature of the labels and facilitates the training process.

The keras.utils.to_categorical function is used to perform this one-hot encoding on the labels y_train and y_test. The second argument, num_classes, specifies the total number of classes in the classification task. In the case of MNIST, num_classes is set to 10 since there are 10 digits (0 through 9).
