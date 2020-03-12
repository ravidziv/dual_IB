"""Find the connection between the different information!"""

from __future__ import absolute_import, division, print_function, unicode_literals
import logging, os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from absl import flags
from absl import app

import numpy as np
import matplotlib.pyplot as plt

from datasets import create_dataset
from utils import store_data, log_summary, load_matrices
import shutil
import os
from information_estimators import get_ixt_all_layers, get_information_all_layers_MINE, get_information_all_layers_clusterd
from information_estimators import  get_information_bins_estimators
from dual_ib import get_information_dual_all_layers, beta_func
FLAGS = flags.FLAGS

flags.DEFINE_string('summary_path','logs/', '')
flags.DEFINE_string('checkpoint_path','log/', '')


flags.DEFINE_integer('x_dim',10, '')
flags.DEFINE_string('nonlin','Relu', '')
flags.DEFINE_integer('num_train', 6000, '')
flags.DEFINE_integer('num_test', 2000, '')
flags.DEFINE_integer('input_shape', 10, '')
flags.DEFINE_integer('y_dim', 2, '')

flags.DEFINE_float('alpha', 0.15, '')
flags.DEFINE_float('lambd', 10.6, '')

flags.DEFINE_float('lr', 1e-2, '')
flags.DEFINE_float('beta', 0.99, '')
flags.DEFINE_integer('num_epochs', 100, '')
flags.DEFINE_integer('batch_per_epoch', 100, '')
flags.DEFINE_integer('num_iterations_to_print', 5, '')
flags.DEFINE_integer('num_of_epochs_inf_labels', 5, '')
flags.DEFINE_integer('num_of_samples', 10, '')
flags.DEFINE_integer('num_of_bins', 30, 'The number of bins for the bins estimator')
flags.DEFINE_integer('batch_size', 128, 'batch size for the main network')

flags.DEFINE_float('noisevar', 1e-1, '')
flags.DEFINE_float('lr_labels', 5e-2, '')
flags.DEFINE_multi_integer('layer_widths', [10, 5], '')
flags.DEFINE_multi_integer('num_of_clusters', [3, 3, 3], 'The widhts of the layers in the random network')

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.FileHandler('log_file'))

def bulid_model(input_shape, y_dim=2, nonlin = 'tanh'):
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Dense(10, input_shape=(input_shape,), activation=nonlin))
  model.add(tf.keras.layers.Dense(10, activation=nonlin))
  model.add(tf.keras.layers.Dense(y_dim))
  return model

def train_step(batch, model , loss_fn, optimizer):
  inputs, targets = batch
  with tf.GradientTape() as tape:
    logits = model(inputs)
    loss_value = loss_fn(targets, logits)
    grads =  tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
  return loss_value


def test_step(model, batch, loss_fn):
  inputs, targets = batch
  logits = model(inputs)
  loss_value = loss_fn(targets, logits)
  return loss_value


def get_linear_information(model, batch_test, entropy_y, num_of_epochs_inf_labels,lr_labels, noisevar):
    itys = get_ity_all_layers(model, batch_test, entropy_y=entropy_y,
                              num_of_epochs=num_of_epochs_inf_labels, lr=lr_labels)

    ixts = get_ixt_all_layers(model=model, x_test=batch_test[0], noisevar=noisevar)
    return ixts, itys

def get_ity_all_layers(model, batch, entropy_y, num_of_epochs, lr=1e-3):
    """Calculate the linear estimator for I(T;Y) (on the top of each layer)"""
    x_test, targets = batch
    itys = [get_iyt([tf.keras.Model(model.inputs, model.layers[layer_indec].output)(x_test), targets], entropy_y=entropy_y,
                    num_of_epochs=num_of_epochs, lr=lr) for layer_indec in range(len(model.layers))]
    return itys

def get_iyt(batch, entropy_y, num_of_epochs, lr=1e-3):
    """Train network to fit logp(y|t)."""
    pred, targets = batch
    model = bulid_model(pred.shape[1], targets.shape[-1])
    optimizer = tf.keras.optimizers.Adam(lr)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    for i in range(num_of_epochs):
        loss_value = train_step(batch, model, loss_fn, optimizer)
    return entropy_y- loss_value


def train(train_ds, test_ds, lr, beta, num_epochs, batch_per_epoch,
          num_iterations_to_print, noisevar, py, py_x, xs, px, A, lambd,
          lr_labels, num_of_clusters, num_of_epochs_inf_labels, beta_func, num_of_samples,
          num_of_bins):
    """Train the model and measure the information."""
    model = bulid_model(FLAGS.input_shape, FLAGS.y_dim)
    optimizer = tf.keras.optimizers.SGD(lr, beta)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    summary_writer = tf.summary.create_file_writer(FLAGS.summary_path)
    #ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    #manager = tf.train.CheckpointManager(ckpt, FLAGS.checkpoint_path, max_to_keep=3)
    #ckpt.restore(manager.latest_checkpoint)
    matrices = load_matrices(model)
    ind = 0
    for epoch in range(num_epochs):
        for batch in train_ds.take(batch_per_epoch):
            loss_value = train_step(batch, model , loss_fn, optimizer)
            if ind % num_iterations_to_print == 0:
                for batch_test in  test_ds.take(1):
                    test_loss_val = test_step(model=model, batch=batch_test, loss_fn=loss_fn)
                    information_bins = get_information_bins_estimators(batch_test, model, num_of_bins=num_of_bins)
                    linear_information = get_linear_information(model, batch_test, py.entropy(), num_of_epochs_inf_labels, lr_labels, noisevar)
                    information_MINE = get_information_all_layers_MINE(model=model, x_test=batch_test[0], y_test=batch_test[1])

                    information_clustered = get_information_all_layers_clusterd(
                        model=model, x_test=batch_test[0], num_of_clusters=num_of_clusters, py_x=py_x, xs=xs, py=py, num_of_samples=num_of_samples)
                    information_dual_ib = get_information_dual_all_layers(model=model, num_of_clusters=num_of_clusters, xs=xs,
                                                    A=A, lambd=lambd, px=px, py=py, beta_func=beta_func)
                    store_data(matrices, loss_value, test_loss_val, linear_information, information_MINE, information_clustered, information_dual_ib, information_bins)
                log_summary(summary_writer, optimizer, epoch, matrices, logger)
            ind += 1

def main(argv):
    #Delete previous runs
    if os.path.isdir(FLAGS.summary_path):
        shutil.rmtree(FLAGS.summary_path)
    train_ds, test_ds, py, py_x, xs, px, A, lambd= create_dataset(FLAGS.num_train, FLAGS.num_test,
                                       FLAGS.x_dim, FLAGS.layer_widths,  FLAGS.nonlin, batch_size=FLAGS.batch_size,
                                       lambd_factor=FLAGS.lambd,
                                       alpha=FLAGS.alpha)
    train(train_ds, test_ds, FLAGS.lr, FLAGS.beta, FLAGS.num_epochs,
          FLAGS.batch_per_epoch,
          FLAGS.num_iterations_to_print, FLAGS.noisevar, py=py, py_x=py_x, xs=xs, px=px, A=A, lambd=lambd,
          lr_labels=FLAGS.lr_labels, num_of_clusters=FLAGS.num_of_clusters,
          num_of_epochs_inf_labels=FLAGS.num_of_epochs_inf_labels, beta_func=beta_func,
          num_of_samples=FLAGS.num_of_samples, num_of_bins=FLAGS.num_of_bins)
    logger.info('Done')


if __name__ == "__main__":
    app.run(main)