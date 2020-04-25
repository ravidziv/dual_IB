"""Models that represent the encoder, decoder and prior of the vib."""
import tensorflow as tf
import tensorflow.keras as keras
tfkl = tf.keras.layers
import tensorflow_probability as tfp
tfd = tfp.distributions


class BasePrior(keras.Model):
    def __init__(self, z_dims = 128):
        super(BasePrior, self).__init__()
        self.z_dim = z_dims
    def call(self, inputs):
        return  tfd.MultivariateNormalDiag(loc=tf.zeros((self.z_dim)))

class BZYPrior(keras.Model):
    def __init__(self, num_claases=10, z_dims = 128):
        super(BZYPrior, self).__init__()
        self.num_classes = num_claases
        self.z_dims = z_dims
        self.net = tf.keras.Sequential([tfkl.Dense(self.z_dims, activation=None)])

    def call(self, inputs):
        """Builds the backwards distribution, b(z|y)."""
        y_onehot = tf.one_hot(inputs, self.num_classes)
        mus =self.net(y_onehot)
        dist = tfd.MultivariateNormalDiag(loc=mus)
        return dist

class BasedEncoder(keras.Model):
    def __init__(self, z_dim=128, h_dim = 1024, activation = 'relu', layer_input_shape =(28, 28, 1)):
        super(BasedEncoder, self).__init__()
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.layer_input_shape=layer_input_shape
        self.activation=activation
        self.net  = tf.keras.Sequential([tfkl.Flatten(input_shape=self.layer_input_shape),
                                         tfkl.Dense(self.h_dim, activation=self.activation),
                                         tfkl.Dense(self.h_dim, activation=self.activation),
                                         tfkl.Dense(2*self.z_dim)
                                         ] )

    def call(self, inputs):
        params = self.net(inputs)
        mu, rho = params[:, :self.z_dim], params[:,self.z_dim:]
        encoding = tfp.layers.DistributionLambda(lambda t: tfd.MultivariateNormalDiag(loc=t[0], scale_diag= tf.math.softplus(t[1])))([mu, rho])
        return encoding

class BaseDecoder(keras.Model):
    def __init__(self, num_of_labels=10):
        super(BaseDecoder, self).__init__()
        self.net =  tf.keras.Sequential(tfkl.Dense(num_of_labels))

    def call(self, inputs):
        net = self.net(inputs)
        return net