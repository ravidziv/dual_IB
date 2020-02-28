
from tensorflow.keras.models import Model
import tensorflow as tf

def mine_loss(args, batch_size):
    t_xy = args[0]
    t_xy_bar = args[1]
    log_term = tf.math.log(tf.cast(batch_size, tf.float32))
    loss = -(tf.reduce_mean(t_xy) - tf.reduce_logsumexp(t_xy_bar) + log_term)
    return loss


class MINE(Model):
    def __init__(self, x_dim=None, y_dim=None, network=None, batch_size=32):
        super(MINE, self).__init__()
        self.model = None
        self.batch_size = batch_size
        self.input_shpe = x_dim +y_dim
        if network is None:
            assert x_dim is not None and y_dim is not None, 'x_dim and y_dim should be both given.'
            self.x_dim = x_dim
            self.y_dim = y_dim
            self.network = self._build_network()
        else:
            assert isinstance(network, Model), 'the network should be defined as a Keras Model class'
            self.network = network

    def call(self, inputs):
        x, y = inputs[:, :-1], inputs[:, -1]
        y_bar_input = tf.random.shuffle(y)  # shuffle y input as y_bar
        ing = tf.concat([x, y[:, None]], axis=1)
        t_xy = self.network(ing)
        ingf = tf.concat([x, y_bar_input[:, None]], axis=1)
        t_xy_bar = self.network(ingf)
        loss = mine_loss([t_xy, t_xy_bar], self.batch_size)
        mutual_information = -loss
        self.add_loss(loss)
        return tf.zeros((10,2), dtype = tf.float64)

    def _build_network(self):
        # build a three-layer fully connected network with 100 units at each layer with ELU activation functions
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(100, input_shape=(self.input_shpe,), activation='elu'))
        model.add(tf.keras.layers.Dense(100, activation='elu'))
        model.add(tf.keras.layers.Dense(100, activation='elu'))
        model.add(tf.keras.layers.Dense(1))
        return model

