"""Implement a  family of variational models - ceb, vib and the dual ib """
import tensorflow as tf
import  tensorflow.keras as keras
import math
import numpy as np
import tensorflow_datasets as tfds
from absl import flags
from absl import app
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np
from VdualIB.models import BasePrior, BaseDecoder, BasedEncoder, BZYPrior
tfkl = tf.keras.layers
dfd = tfp.distributions
FLAGS = flags.FLAGS

flags.DEFINE_integer('z_dim', 128, 'The dimension of the hidden space')
flags.DEFINE_integer('num_of_labels', 10, 'The number of labels')
flags.DEFINE_integer('h_dim', 10, 'the dimension of the hidden layers of the encoder')
flags.DEFINE_integer('batch_size', 128, 'For training')
flags.DEFINE_integer('num_of_epochs', 25, 'For training')
flags.DEFINE_string('activation', 'relu', 'Activation of the encoder layers')
flags.DEFINE_multi_enum('run_model', 'vib', ['vib', 'ceb','dual_ib'],  'Which model to run')
flags.DEFINE_float('initial_lr', 1e-3, 'The lr for the train')
flags.DEFINE_float('beta', 1e-3, 'beta value for the loss function')


@tf.function
def loss_func(labels, logits, z, encoding, prior, beta, num_of_labels=10, scale= 1e-1, loss_func_inner=None):
    cyz = dfd.Categorical(logits=logits)
    labels_one_hot = tf.one_hot(labels, num_of_labels)
    #labels_one_hot-= 1./num_of_labels
    scale_param = scale*tf.ones((labels.shape[0], num_of_labels))
    labels_dist = dfd.MultivariateNormalDiag(loc=tf.cast(labels_one_hot, tf.float32),
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
        self.add_metric(math.log(num_of_labels) - hyz, name='IZY', aggregation='mean')
        self.add_metric(hzy - hzx, name='IZX', aggregation='mean')
        self.add_metric(math.log(num_of_labels) - hyhatz, name='IZYhat', aggregation='mean')
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


def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

@tf.function
def loss_func_ib(hyz, hzy, hzx, hyhatz, beta):
    info_loss =hzy - hzx
    total_loss = hyz + beta * info_loss
    return total_loss


@tf.function
def loss_func_dual_ib(hyz, hzy, hzx, hyhatz, beta):
    info_loss =hzy - hzx
    total_loss = hyhatz + beta * info_loss
    return total_loss


def main(argv):
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    # Create Data
    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(FLAGS.batch_size, drop_remainder=True)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(FLAGS.batch_size, drop_remainder=True)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    # Create encoder, decoder and the model
    if FLAGS.run_model[0] =='ceb' or FLAGS.run_model[0] == 'vib':
        loss_func_inner = loss_func_ib
        if FLAGS.run_model =='ceb':
            prior = BZYPrior(z_dims=FLAGS.z_di)
        else:
            prior = BasePrior(z_dims=FLAGS.z_dim)
    elif FLAGS.run_model[0] == 'dual_ib':
        loss_func_inner = loss_func_dual_ib
        prior = BasePrior(z_dims=FLAGS.z_dim)

    encoder = BasedEncoder(z_dim=FLAGS.z_dim, h_dim=FLAGS.h_dim, activation=FLAGS.activation,
                           layer_input_shape=(28, 28, 1))
    decoder = BaseDecoder(num_of_labels=FLAGS.num_of_labels)

    model = VariationalNetwork(beta=FLAGS.beta, encoder=encoder, decoder=decoder, prior=prior,
                               loss_func_inner=loss_func_inner)
    class_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    opt = tf.keras.optimizers.Adam(FLAGS.initial_lr)
    model.compile(optimizer=opt, loss=class_loss_fn, metrics=['acc'])
    # model.run_eagerly = True
    # Train
    model.fit(ds_train, epochs=FLAGS.num_of_epochs, verbose=2, validation_data=ds_test)


if __name__ == '__main__':
    if __name__ == "__main__":
        app.run(main)
