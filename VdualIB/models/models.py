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
        self.net = tfp.layers.DistributionLambda(lambda t: tfd.MultivariateNormalDiag(loc=1 * tf.ones((self.z_dim))))
    def call(self, inputs):
        return  self.net(0)


class BaseLabelsModel(keras.Model):
    def __init__(self, num_claases=10, z_dims=128, confusion_matrix=None):
        super(BaseLabelsModel, self).__init__()
        self.num_classes = num_claases
        self.z_dims = z_dims
        self.confusion_matrix = confusion_matrix
        # self.net = tf.keras.Sequential(
        #    [tfkl.InputLayer(input_shape=(self.num_classes,)), tfkl.Dense(2*self.num_classes, activation=None), tfkl.Dense(2*self.num_classes, activation=None)])

    def call(self, inputs):
        """Builds the backwards distribution, b(z|y)."""
        y_onehot = tf.one_hot(inputs, self.num_classes)
        # params = self.net(y_onehot)
        # mu, rho = params[:, :self.num_classes], params[:,self.num_classes:]

        # encoding = tfp.layers.DistributionLambda(lambda t: tfd.MultivariateNormalDiag(loc=t[0], scale_diag= len()*tf.ones((y_onehot.shape))))([y_onehot])
        encoding = tfp.layers.DistributionLambda(lambda t: tfd.MultivariateNormalTriL(loc=t[0], scale_tril=
        1e-6 + 1e-1 * tf.linalg.cholesky(1e-4 + tf.cast(self.confusion_matrix.T, tf.float32))))([y_onehot])
        return encoding


class BZYPrior(keras.Model):
    def __init__(self, num_classes=10, h_dims=500, z_dims=128):
        super(BZYPrior, self).__init__()
        self.num_classes = num_classes
        self.z_dims = z_dims
        self.net = tf.keras.Sequential(
            [tfkl.InputLayer(input_shape=(self.num_classes,)), tfkl.Dense(h_dims, activation='relu'),
             tfkl.Dense(2 * self.z_dims, activation=None)])

    # def call(self, inputs):
    #    """Builds the backwards distribution, b(z|y)."""
    #    mus =self.net(y_onehot)
    #    dist = tfd.MultivariateNormalDiag(loc=mus)
    #    return dist

    def call(self, inputs):
        y_onehot = tf.one_hot(inputs, self.num_classes)

        params = self.net(y_onehot)
        mu, rho = params[:, :self.z_dims], params[:, self.z_dims:]
        encoding = tfp.layers.DistributionLambda(
            lambda t: tfd.MultivariateNormalDiag(loc=t[0], scale_diag=1e-2 + tf.math.softplus(t[1])))([mu, rho])
        return encoding


def build_default_cov_net(layer_input_shape=(32, 32, 3)):
    """Build a simple 3 layers covnet."""
    net = tf.keras.Sequential()
    net.add(tfkl.Conv2D(32, (3, 3), activation='relu', input_shape=layer_input_shape))
    net.add(tfkl.MaxPooling2D((2, 2)))
    net.add(tfkl.Conv2D(64, (3, 3), activation='relu'))
    net.add(tfkl.MaxPooling2D((2, 2)))
    net.add(tfkl.Conv2D(64, (3, 3), activation='relu'))
    net.add(tfkl.Flatten())
    return net


def build_default_FC_net(z_dim, h_dim, layer_input_shape, activation):
    """Return 2 layers FC network"""
    net = tf.keras.Sequential([tfkl.Flatten(input_shape=layer_input_shape),
                                    tfkl.Dense(h_dim, activation=activation),
                                    tfkl.Dense(h_dim, activation=activation),
                                    ])
    return net
class BasedEncoder(keras.Model):
    def __init__(self, z_dim=128, h_dim = 1024, activation = 'relu', layer_input_shape =(28, 28, 1), net=None):
        super(BasedEncoder, self).__init__()
        self.z_dim = z_dim
        if net:
            self.net = net
        else:
            self.net = build_default_FC_net(z_dim, h_dim, layer_input_shape, activation)


    def call(self, inputs):
        params = self.net(inputs)
        # mu, rho = params[:, :self.z_dim], params[:, self.z_dim:]
        # encoding = tfp.layers.DistributionLambda(
        #    lambda t: tfd.MultivariateNormalDiag(loc=t[0], scale_diag=1e-3 + tf.math.softplus(t[1])))([mu, rho])
        return params

class BaseDecoder(keras.Model):
    def __init__(self, latent_dim=2048, h_dim=1000, num_of_labels=10, weight_decay=0.0005 ):
        super(BaseDecoder, self).__init__()
        self.weight_decay = weight_decay
        self.net = tf.keras.Sequential(
            [tfkl.InputLayer(input_shape=(latent_dim,)), tfkl.Dense(h_dim), tfkl.Dense(num_of_labels,
                                                                                       kernel_initializer='he_normal',
                                                                                       kernel_regularizer=l2(self.weight_decay),
                                                                                       use_bias=False),
             # tfkl.Softmax()
             ])
    def call(self, inputs):
        net = self.net(inputs)
        return net
