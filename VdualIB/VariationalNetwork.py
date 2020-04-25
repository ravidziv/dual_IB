import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
tfd = tfp.distributions


@tf.function
def loss_func(labels, logits, z, encoding, prior, beta, num_of_labels=10, scale= 1e-1, loss_func_inner=None):
    cyz = tfd.Categorical(logits=logits)
    labels_one_hot = tf.one_hot(labels, num_of_labels)
    #labels_one_hot-= 1./num_of_labels
    scale_param = scale*tf.ones((labels.shape[0], num_of_labels))
    labels_dist = tfd.MultivariateNormalDiag(loc=tf.cast(labels_one_hot, tf.float32),
                                             scale_diag=scale_param)
    hyhatz = -labels_dist.log_prob(logits)
    hyz = -cyz.log_prob(labels)
    hzx = -encoding.log_prob(z)
    hzy = -prior.log_prob(z)
    total_loss = tf.reduce_mean(loss_func_inner(hyz, hzy, hzx, hyhatz, beta))
    losses = [hzy,hzx ,hyz,hyhatz ]
    return total_loss, losses


class VariationalNetwork(keras.Model):
    def __init__(self, beta, encoder, decoder, prior, loss_func_inner):
        super(VariationalNetwork, self).__init__()
        self.beta = beta
        self.loss_func_inner=loss_func_inner
        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior

    def call(self, x):
        encoding = self.encoder(x)
        z = encoding.sample()
        cyz = self.decoder(z)
        return encoding, cyz, z


    def write_metrices(self, y, logits, losses, total_loss, num_of_labels=10 ):
        hzy, hzx, hyz, hyhatz = losses
        self.compiled_metrics.update_state(y, logits)
        self.compiled_loss(
            tf.one_hot(y, num_of_labels), logits)
        self.add_metric(tf.math.log(tf.cast(num_of_labels, tf.float32)) - hyz, name='IZY', aggregation='mean')
        self.add_metric(hzy - hzx, name='IZX', aggregation='mean')
        self.add_metric(tf.math.log(tf.cast(num_of_labels, tf.float32)) - hyhatz, name='IZYhat', aggregation='mean')
        self.add_metric(total_loss, name='bound', aggregation='mean')

    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        encoding, cyz, z = self(x)
        prior = self.prior(y)
        total_loss, losses = loss_func(y, cyz, z, encoding, prior, self.beta, loss_func_inner=self.loss_func_inner)
        self.write_metrices(y, cyz, losses, total_loss)
        return {m.name: m.result() for m in self.metrics}


    def train_step(self, input):
        x, y = input
        with tf.GradientTape() as tape:
            encoding, cyz, z = self(x)
            prior = self.prior(y)
            total_loss, losses = loss_func(y, cyz, z, encoding, prior, self.beta, loss_func_inner=self.loss_func_inner)
        self.write_metrices(y, cyz, losses, total_loss)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {m.name: m.result() for m in self.metrics}

