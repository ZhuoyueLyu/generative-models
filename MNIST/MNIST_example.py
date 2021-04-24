import tensorflow as tf
from keras.datasets import mnist
from matplotlib import pyplot

(train_X, train_label), _ = mnist.load_data()

print('X shape: ' + str(train_X.shape))
print('Label shape: ' + str(train_label.shape))

idx = 0
print("X[{}]:\n{}".format(idx, train_X[idx, :, :]))
print("Label[{}]: {}".format(idx, train_label[idx]))

alpha = tf.random.uniform((60000, 1), 0, 0.5)
print(alpha)