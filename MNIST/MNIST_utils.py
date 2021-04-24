"""
Utility functions.  
Some codes are based on websites below... 
"""
from keras.datasets import mnist

def get_mnist_data():
    (train_X, train_label), _ = mnist.load_data()
    return train_X

def get_mnist_label():
    (train_X, train_label), _ = mnist.load_data()
    return train_label
