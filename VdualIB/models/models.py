"""Models that represent the encoder, decoder and prior of the vib."""
import tensorflow as tf
import tensorflow.keras as keras
tfkl = tf.keras.layers
import tensorflow_probability as tfp
from tensorflow.keras.regularizers import l2

tfd = tfp.distributions


class BasePrior(keras.Model):
    def __init__(self, z_dims = 128):
        super(BasePrior, self).__init__()
        self.z_dim = z_dims
        self.net = tfp.layers.DistributionLambda(lambda t: tfd.MultivariateNormalDiag(loc=3*tf.ones((self.z_dim))))
    def call(self, inputs):
        return  self.net(0)

class BZYPrior(keras.Model):
    def __init__(self, num_claases=10, z_dims = 128):
        super(BZYPrior, self).__init__()
        self.num_classes = num_claases
        self.z_dims = z_dims
        self.net = tf.keras.Sequential([tfkl.InputLayer(input_shape=(self.num_claases,)), tfkl.Dense(self.z_dims, activation=None)])

    def call(self, inputs):
        """Builds the backwards distribution, b(z|y)."""
        y_onehot = tf.one_hot(inputs, self.num_classes)
        mus =self.net(y_onehot)
        dist = tfd.MultivariateNormalDiag(loc=mus)
        return dist


def build_default_net(z_dim, h_dim, layer_input_shape, activation ):
    net = tf.keras.Sequential([tfkl.Flatten(input_shape=layer_input_shape),
                                    tfkl.Dense(h_dim, activation=activation),
                                    tfkl.Dense(h_dim, activation=activation),
                                    tfkl.Dense(2 * z_dim)
                                    ])
    return net
class BasedEncoder(keras.Model):
    def __init__(self, z_dim=128, h_dim = 1024, activation = 'relu', layer_input_shape =(28, 28, 1), net=None):
        super(BasedEncoder, self).__init__()
        self.z_dim = z_dim
        if net:
            self.net = net
        else:
            self.net = build_default_net(z_dim, h_dim, layer_input_shape, activation)


    def call(self, inputs):
        params = self.net(inputs)
        mu, rho = params[:, :self.z_dim], params[:,self.z_dim:]
        encoding = tfp.layers.DistributionLambda(lambda t: tfd.MultivariateNormalDiag(loc=t[0], scale_diag= tf.math.softplus(t[1])))([mu, rho])
        return encoding

class BaseDecoder(keras.Model):
    def __init__(self, latent_dim=2048, num_of_labels=10,weight_decay=0.0005 ):
        super(BaseDecoder, self).__init__()
        self.weight_decay = weight_decay
        self.net =  tf.keras.Sequential([tfkl.InputLayer(input_shape=(latent_dim,)),tfkl.Dense(num_of_labels,
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(self.weight_decay),
                  use_bias=False)])
    def call(self, inputs):
        net = self.net(inputs)
        return net