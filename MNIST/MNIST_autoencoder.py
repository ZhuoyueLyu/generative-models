"""
Trying autoencoder using MNIST. 
Some codes are based on websites below... 

https://www.tensorflow.org/tutorials/
https://towardsdatascience.com/implementing-an-autoencoder-in-tensorflow-2-0-5e86126e9f7
"""

import tensorflow as tf 
from tensorflow.keras.layers import Dense, Flatten, Conv2D 
from tensorflow.keras import Model 
from MNIST_utils import get_mnist_data

class ACAI_autoencoder(Model):
    def __init__(self, hidden_dim=500, latent_dim=2):
        super().__init__()
        self.encoder1 = Dense(hidden_dim)
        self.encoder2 = Dense(latent_dim)
        self.decoder1 = Dense(hidden_dim)
        self.decoder2 = Dense(784)

    def encode(self, input_tensor):
        x = self.encoder1(input_tensor)
        x = tf.nn.relu(x)
        x = self.encoder2(x)
        return x

    def decode(self, input_tensor):
        x = self.decoder1(input_tensor)
        x = tf.nn.relu(x)
        x = self.decoder2(x)
        return x 

    def call(self, input_tensor):
        x = self.encode(input_tensor)
        x = self.decode(x)
        return x
    
def autoencoder_loss(autoencoder, x):
    x_out = autoencoder(x)
    return tf.reduce_mean(tf.square(tf.subtract(x_out, x)))

def train_autoencoder_step(autoencoder, x):
    with tf.GradientTape() as tape:
        loss = autoencoder_loss(autoencoder, x)
    return loss, tape.gradient(loss, autoencoder.trainable_variables)

### Parameters
hidden_dim = 100
lr = 1e-3
epochs = 5
save=True 
load=True
path="./mnist_autoencoder_model/"

# Getting data 
data = tf.cast(get_mnist_data(), tf.float32)
data = tf.reshape(data, [data.shape[0], data.shape[1] * data.shape[2]])
print("Data shape: {}".format(data.shape))

# Making model 
autoencoder = ACAI_autoencoder(hidden_dim=hidden_dim)
if load:
    autoencoder.load_weights(path)
test_out = autoencoder(data)
print("Autoencoder output shape: {}".format(test_out.shape))

# Create optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
loss, grad = train_autoencoder_step(autoencoder, data)

print("Step: Initial, Initial Loss: {}".format(
    loss.numpy()
))

for i in range(epochs):
    loss, grad = train_autoencoder_step(autoencoder, data)
    print("Step: {}, Loss: {}".format(
        optimizer.iterations.numpy(),
        loss.numpy()
    ))

    optimizer.apply_gradients(zip(grad, autoencoder.trainable_variables))

if save:
    autoencoder.save_weights(path)