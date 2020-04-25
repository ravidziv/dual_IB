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
from VdualIB.VariationalNetwork import VariationalNetwork
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
flags.DEFINE_float('initial_lr', 1e-4, 'The lr for the train')
flags.DEFINE_float('beta', 1e-3, 'beta value for the loss function')


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
