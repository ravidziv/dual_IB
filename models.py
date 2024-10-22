import tensorflow_probability as tfp
from tensorflow.keras import activations
from tensorflow.keras.regularizers import l2
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
tfkl = tf.keras.layers

tfd = tfp.distributions
import tensorflow.keras as keras

class BasePrior(keras.Model):
    def __init__(self, z_dims=128):
        super(BasePrior, self).__init__()
        self.z_dim = z_dims
        self.net = tfp.layers.DistributionLambda(
            lambda t: tfd.MultivariateNormalDiag(scale_identity_multiplier=1, loc=tf.zeros((self.z_dim))))

    def call(self, inputs):
        return self.net(0)


class BaseDecoder(keras.Model):
    def __init__(self, latent_dim=2048, h_dim=1000, num_of_labels=10, weight_decay=1e-5):
        super(BaseDecoder, self).__init__()
        self.weight_decay = weight_decay
        self.net = tf.keras.Sequential(
            [tfkl.InputLayer(input_shape=(latent_dim,)), tfkl.Dense(h_dim, activation=activations.gelu),
             tf.keras.layers.LayerNormalization(),
             tfkl.Dense(num_of_labels,
                        kernel_initializer='he_normal',
                        kernel_regularizer=l2(
                            self.weight_decay),
                        use_bias=False),
             # tfkl.Softmax()
             ])

    def call(self, inputs):
        net = self.net(inputs)
        return net


def generate_tril_dist(params, z_dim):
    mu, rho = params[:, :z_dim], params[:, z_dim:z_dim * 2]
    encoding = tfp.layers.DistributionLambda(
        lambda t: tfp.distributions.MultivariateNormalDiag(loc=t[0], scale_diag=tf.math.softplus(t[1] - 5)
                                                           ))([mu, rho])
    return encoding


class VariationalNetwork(keras.Model):
    def __init__(self, number_of_samples=1, encoder=None, decoder=None, marginal=None, beta=0.5, z_dim=100):
        super(VariationalNetwork, self).__init__()
        self.number_of_samples = number_of_samples
        self.encoder = encoder
        self.decoder = decoder
        self.marginal = marginal
        self.beta = beta
        self.z_dim = z_dim

    def encode(self, x):
        """Take an image and return a two dimensional distribution."""
        params = self.encoder(x)
        return generate_tril_dist(params, self.z_dim)

    def decode(self, z_samps):
        """Given a sampled representation, predict the class."""
        logits = self.decoder(z_samps)
        return tfd.Categorical(logits=logits)

    def call(self, x):
        z_dist = self.encode(x)
        z_samp = tf.reduce_mean(z_dist.sample(self.number_of_samples), axis=0)
        y_dist = self.decode(z_samp)
        return y_dist, z_dist, z_samp

    def train_step(self, inputs):
        x, y = inputs
        with tf.GradientTape() as tape:
            yhat_dist, z_dist, z_samps = self(x, training=True)
            logits = yhat_dist.logits

            total_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_pred=logits, y_true=y)
            lps = yhat_dist.log_prob(y)
            class_err = -tf.reduce_mean(lps)
            rate = tf.reduce_mean(z_dist.log_prob(z_samps) - self.marginal.log_prob(z_samps))
            loss = class_err + self.beta * rate

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.compiled_metrics.update_state(y, logits)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, inputs):
        x, y = inputs
        yhat_dist, z_dist, z_samps = self(x, training=False)
        logits = yhat_dist.logits
        self.compiled_metrics.update_state(y, logits)
        return {m.name: m.result() for m in self.metrics}
