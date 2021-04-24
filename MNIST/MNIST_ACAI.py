"""
Trying ACAI using MNIST. 
Some codes are based on websites below... 

https://www.tensorflow.org/tutorials/
https://towardsdatascience.com/implementing-an-autoencoder-in-tensorflow-2-0-5e86126e9f7
https://github.com/baohq1595/aae-experiment/blob/master/src/notebook/acai_mnist_tf.ipynb
"""
import tensorflow as tf 
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input
from tensorflow.keras import Model 
from tensorflow.keras.models import Sequential
from MNIST_utils import get_mnist_data

mnist_shape = 784 

class ACAI_autoencoder(Model):
    def __init__(self, hidden_dim=500, latent_dim=2):
        super().__init__()
        self.autoencoder_lambda = 100
        self.encoder = Sequential([
            Dense(hidden_dim, activation='relu', input_shape=(mnist_shape,)),
            Dense(latent_dim)
        ])
        self.decoder = Sequential([
            Dense(hidden_dim, activation='relu', input_shape=(latent_dim,)),
            Dense(mnist_shape)
        ])
        self.critic = Sequential([
            Dense(hidden_dim, activation='relu', input_shape=(mnist_shape,)),
            Dense(int(hidden_dim / 5), activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        x = Input(shape=mnist_shape)
        z = self.encoder(x)
        x_out = self.decoder(z)
        
        self.autoencoder = Model(x, x_out)

def autoencoder_loss(acai, x, z, critic_alpha):
    x_out = acai.decoder(z)
    recon_loss = tf.reduce_mean(tf.square(tf.subtract(x_out, x))) 
    critic_loss = tf.reduce_mean(critic_alpha)
    return recon_loss + acai.autoencoder_lambda*critic_loss

def critic_loss(alpha, critic_alpha):
    return tf.reduce_mean(tf.square(alpha - critic_alpha)) 

def train_ACAI_step(acai, x, alpha):
    with tf.GradientTape() as auto_tape, tf.GradientTape() as critic_tape:
        z = acai.encoder(x)

        mix_z = tf.multiply(z, alpha) + tf.multiply(z[::-1], 1-alpha)
        critic_alpha = acai.critic(acai.decoder(mix_z))

        auto_loss = autoencoder_loss(acai, x, z, critic_alpha)
        crit_loss = critic_loss(alpha, critic_alpha)

    return auto_loss, auto_tape.gradient(auto_loss, acai.autoencoder.trainable_variables),\
        crit_loss, critic_tape.gradient(crit_loss, acai.critic.trainable_variables)

if __name__ == "__main__":
    ### Parameters
    hidden_dim = 400
    latent_dim = 20
    lr = 5e-4
    epochs = 70
    save=True 
    load=True
    path="./mnist_acai/"

    # Getting data 
    data = tf.cast(get_mnist_data(), tf.float32)
    data = tf.reshape(data, [data.shape[0], data.shape[1] * data.shape[2]])
    print("Data shape: {}".format(data.shape))

    # Making model 
    acai = ACAI_autoencoder(hidden_dim=hidden_dim, latent_dim=latent_dim)
    if load:
        acai.decoder.load_weights(path+'decoder/')
        acai.encoder.load_weights(path+'encoder/')
        acai.critic.load_weights(path+'critic/')
    #test_out = autoencoder(data)
    #print("Autoencoder output shape: {}".format(test_out.shape))

    # Create optimizer
    auto_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    crit_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    alpha = tf.random.uniform((60000, 1), 0, 0.5)
    autoloss, autograd, critloss, crit_grad = train_ACAI_step(acai, data, alpha)

    print("Step: Initial, Autoencoder Loss: {}    Critic Loss: {}".format(
        autoloss.numpy(), critloss.numpy()
    ))

    # Training loop 
    for i in range(epochs):
        alpha = tf.random.uniform((60000, 1), 0, 0.5)
        autoloss, autograd, critloss, crit_grad = train_ACAI_step(acai, data, alpha)

        print("Step: {}, Autoencoder Loss: {}    Critic Loss: {}".format(
            auto_optimizer.iterations.numpy(), autoloss.numpy(), critloss.numpy()
        ))

        auto_optimizer.apply_gradients(zip(autograd, acai.autoencoder.trainable_variables))
        crit_optimizer.apply_gradients(zip(crit_grad, acai.critic.trainable_variables))

    if save:
        acai.decoder.save_weights(path+'decoder/')
        acai.encoder.save_weights(path+'encoder/')
        acai.critic.save_weights(path+'critic/')
