from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow import keras
import tensorflow as tf
from  tensorflow_probability import distributions as tfd
from absl import flags
from absl import app
import logging

import numpy as np
import utils
import matplotlib.pyplot as plt
import itertools
import numpy.random as npr
from scipy.special import logsumexp, softmax
from sklearn.metrics import pairwise_distances
import scipy
import scipy as sp
from sklearn import cluster
from datasets import create_dataset
from sklearn import mixture
import shutil
import os

FLAGS = flags.FLAGS

flags.DEFINE_string('summary_path','logs/', '')
flags.DEFINE_string('checkpoint_path','log/', '')


flags.DEFINE_integer('x_dim',10, '')
flags.DEFINE_string('nonlin','Relu', '')
flags.DEFINE_integer('num_train', 2000, '')
flags.DEFINE_integer('num_test', 2000, '')
flags.DEFINE_integer('input_shape', 10, '')
flags.DEFINE_integer('y_dim', 2, '')

flags.DEFINE_float('alpha', 0.15, '')
flags.DEFINE_float('lambd', 4.6, '')

flags.DEFINE_float('lr', 1e-2, '')
flags.DEFINE_float('beta', 0.99, '')
flags.DEFINE_integer('num_epochs', 100, '')
flags.DEFINE_integer('batch_per_epoch', 10, '')
flags.DEFINE_integer('num_iterations_to_print', 5, '')
flags.DEFINE_float('noisevar', 1e-1, '')

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.FileHandler('log_file'))

def bulid_model(input_shape, y_dim=2):
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Dense(10, input_shape=(input_shape,), activation='relu'))
  model.add(tf.keras.layers.Dense(10, activation='relu'))
  model.add(tf.keras.layers.Dense(y_dim))
  return model


def get_ity_all_layers(model, batch, entropy_y, num_of_epochs, lr=1e-3):
    x_test, targets = batch
    itys = [get_iyt([tf.keras.Model(model.inputs, model.layers[layer_indec].output)(x_test), targets], entropy_y=entropy_y,
                    num_of_epochs=num_of_epochs, lr=lr) for layer_indec in range(len(model.layers))]
    return itys

def get_iyt(batch, entropy_y, num_of_epochs, lr=1e-3):
    pred, targets = batch
    model = bulid_model(pred.shape[1], targets.shape[-1])
    optimizer = tf.keras.optimizers.Adam(lr)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    for i in range(num_of_epochs):
        loss_value = train_step(batch, model, loss_fn, optimizer)
        #print (i, loss_value)
    return entropy_y- loss_value

def train_step(batch, model , loss_fn, optimizer):
  inputs, targets = batch
  with tf.GradientTape() as tape:
    logits = model(inputs)
    loss_value = loss_fn(targets, logits)
    grads =  tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
  return loss_value


def GMM_entropy(dist, var, d, bound='upper', weights =None):
    # computes bounds for the entropy of a homoscedastic Gaussian mixture model [Kolchinsky, 2017]
    # dist: a matrix of pairwise distances
    # log_var: the log-variance of the mixture components
    # d: number of dimensions
    # n: number of mixture components
    pac = 0.1
    n = dist.shape[0]

    if bound is 'upper':
        dist_norm = - dist / (2.0 * var)  # uses the KL distance
    elif bound is 'lower':
        dist_norm = - dist / (8.0 * var)  # uses the Bhattacharyya distance
    const = 0.5 * d * np.log(2.0 * np.pi * np.exp(1.0) * var) + np.log(n) +pac
    if weights is not None:
      h = np.average(const, weights = weights) - np.average(logsumexp(dist_norm, 1, b=weights), weights=weights)
    else:
      h = np.average(const) - np.mean(logsumexp(dist_norm, 1))
    return h

def gaussian_entropy_np(d, var):
    # Entropy of a Gaussian distribution with 'd' dimensions and log variance 'log_var'
    h = 0.5 * d * (np.log(2.0 * np.pi * np.exp(1)) + np.log(var))
    return h



def get_ixt(inputs, noisevar= 1e-1):
  inputs = inputs.numpy().astype(np.float64)
  input_dim = inputs.shape[1]
  H_T_given_X = gaussian_entropy_np(input_dim, noisevar)
  dist_matrix = pairwise_distances(inputs)
  H_T_lb        = GMM_entropy(dist_matrix, noisevar, input_dim, 'lower')
  Ixt_lb        = H_T_lb - H_T_given_X
  H_T           = GMM_entropy(dist_matrix, noisevar, input_dim, 'upper')
  Ixt           = H_T - H_T_given_X  # nonlinear IB upper bound on I(X;T)
  return Ixt

def cluster_data(data, py_x, n_clusters):
  k_means = cluster.KMeans(n_clusters=n_clusters, n_init=10)
  k_means.fit(data)
  values = k_means.cluster_centers_.squeeze()
  labels = k_means.labels_
  var_clusters = []
  clusterd_data = np.array(data)
  px_t = np.zeros((data.shape[0], values.shape[0]))
  cs = []
  for cluster_index in range(n_clusters):
    xs_cluster = data[labels ==cluster_index,:]
    clusterd_data[labels ==cluster_index, :] = values[cluster_index]
    var_cluster = np.var(xs_cluster - values[cluster_index])
    expt = np.exp(-(1/(2*var_cluster))*(np.linalg.norm(xs_cluster - values[cluster_index], axis=1)**2))
    px_t[labels ==cluster_index,cluster_index] = expt
    if np.sum(expt) ==0:
      px_t[labels ==cluster_index,cluster_index] = 1
    elif np.isnan(np.sum(expt)) :
      px_t[labels ==cluster_index,cluster_index] = 1
    cs.append(len(xs_cluster))
    var_clusters.append(var_cluster)
  px_t = np.array(px_t)
  indexs = ~np.any(px_t>0, axis=0)
  in_w = np.where(indexs)
  px_t[in_w,in_w]=1
  px_t = np.copy(px_t) / np.sum(np.copy(px_t), axis=0)
  py_t = np.sum(py_x.T[:,  :,None]*px_t[ None,:, :], axis=1)
  clusterd_pred = np.zeros((data.shape[0],py_t.shape[0] ))
  for cluster_index in range(n_clusters):
    clusterd_pred[labels ==cluster_index, :] = py_t[:, cluster_index]
  cs = np.array(cs)/np.sum(np.array(cs))
  var_clusters = np.array(var_clusters)
  return cs, var_clusters, values, labels ,px_t, clusterd_pred, py_t

def get_ixt_clustered(data, num_of_clusters, num_of_samples):
    """Calculate ixt after clustring to mixture of gaussian"""
    clf = mixture.GaussianMixture(n_components=num_of_clusters, covariance_type='tied')
    clf.fit(data)
    qxt = tfd.MultivariateNormalTriL(loc=clf.means_, scale_tril=tf.linalg.cholesky(clf.covariances_))
    ixt_array = 0
    for i in range(num_of_samples):
        sample_x = qxt.sample()
        log_pxt = qxt.log_prob(sample_x[:, None])
        diag_xt = np.diag(log_pxt)
        K = log_pxt.shape[1]
        h_negative_yz = tf.reduce_logsumexp(log_pxt, axis=1) - np.log(K)
        ixt = diag_xt - h_negative_yz
        ixt_array += tf.reduce_mean(ixt)
    ixt_array /= num_of_samples
    return ixt_array

def get_information_dual(data, num_of_clusters, A, lambd, x_entopy, y_entopy):
    """Calculate ixt after clustring to mixture of gaussian"""
    clf = mixture.GaussianMixture(n_components=num_of_clusters, covariance_type='tied')
    clf.fit(data)
    qxt = tfd.MultivariateNormalTriL(loc=clf.means_, scale_tril=tf.linalg.cholesky(clf.covariances_))
    pt = clf.weights_
    beta = 1/ (np.trace(clf.covariances_))

    px_t = []
    for t_index in range(num_of_clusters):

        px_t.append(qxt[t_index].prob(tf.cast(data, tf.float64)))
    px_t = tf.transpose(tf.stack(px_t))
    ixt = utils.calc_IXT_p(pt, px_t.numpy(), A.numpy().T, lambd.numpy().T, beta, x_entopy)
    ity = utils.calc_IYT_p(pt, px_t.numpy(), A.numpy().T, lambd.numpy().T, y_entopy)
    return ixt, ity


def get_information_dual_all_layers(model, num_of_clusters=[10, 10, 10], xs=None,
                                    A=None, lambd=None, px=None, py=None):
    itys = [get_information_dual(tf.keras.Model(model.inputs, model.layers[layer_indec].output)(xs),
                                 num_of_clusters=num_of_clusters[layer_indec],  A=A, lambd=lambd, x_entopy=px.entropy(),
                                 y_entopy=py.entropy()) for layer_indec in range(len(model.layers))]
    return itys

def get_iyt_clustered(data, num_of_clusters, py_x, py):
    """Calculate ixt after clustring to mixture of gaussian"""
    clf = mixture.GaussianMixture(n_components=num_of_clusters, covariance_type='tied')
    clf.fit(data)
    qxt = tfd.MultivariateNormalTriL(loc=clf.means_, scale_tril=tf.linalg.cholesky(clf.covariances_))
    log_py_ts = []
    log_py_x = tf.math.log(py_x)

    for t_index in range(num_of_clusters):
        logpx_t = qxt[t_index].log_prob(tf.cast(data, tf.float64))
        log_py_t =  tf.reduce_logsumexp(tf.cast(log_py_x, tf.float64)*tf.cast(logpx_t[:, None], tf.float64), axis=0)
        log_py_t /=np.sum(log_py_t)
        log_py_ts.append(log_py_t)
    hyt = tf.reduce_mean(tf.cast(py._probs, tf.float64)[:, None]*tf.transpose(tf.stack(log_py_ts)))
    return tf.cast(py.entropy(), tf.float64) -hyt

def get_ixt_all_layers_clusters(model, x_test, num_of_clusters=[10, 10, 10],  num_of_samples = 100):
    ixts = [get_ixt_clustered(tf.keras.Model(model.inputs, model.layers[layer_indec].output)(x_test), num_of_clusters=num_of_clusters[layer_indec], num_of_samples=num_of_samples) for layer_indec in range(len(model.layers))]
    return ixts

def get_ity_all_layers_clusters(model, x_test, num_of_clusters=[10, 10, 10], py=None, py_x=None, xs=None):
    itys = [get_iyt_clustered(tf.keras.Model(model.inputs, model.layers[layer_indec].output)(xs), num_of_clusters=num_of_clusters[layer_indec], py_x=py_x, py=py) for layer_indec in range(len(model.layers))]
    return itys

def get_ixt_all_layers(model, x_test, noisevar=1e-3):
  #Get I(X;T) for each layer
  ixts =  [get_ixt(tf.keras.Model(model.inputs, layer.output)(x_test), noisevar= noisevar) for layer in model.layers]
  return ixts

def test_step(model, batch, loss_fn):
  inputs, targets = batch
  logits = model(inputs)
  loss_value = loss_fn(targets, logits)
  return loss_value


def log_summary(summary_writer, optimizer, epoch, matrices):
    message = 'Epoch {}, '.format(epoch+1)
    with summary_writer.as_default():
        for matric in matrices:
            tf.summary.scalar(matric.name, matric.result(), step = optimizer.iterations)
            message +='{} {:0.3f}, '.format(matric.name, matric.result())
            matric.reset_states()
        logger.info(message)


def train(model, train_ds, test_ds, lr, beta, num_epochs, batch_per_epoch,
          num_iterations_to_print, noisevar, py, py_x, xs, px, A, lambd):
    summary_writer = tf.summary.create_file_writer(FLAGS.summary_path)

    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.lr, beta_1=FLAGS.beta)

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, FLAGS.checkpoint_path, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)

    train_loss = tf.keras.metrics.Mean(name = 'Train loss')
    test_loss = tf.keras.metrics.Mean(name = 'Test loss')
    test_ixt_bound = [tf.keras.metrics.Mean(name = r"Test I(X;T_{})".format(i)) for i in range(len(model.layers))]
    test_ity_bound = [tf.keras.metrics.Mean(name = r"Test I(Y;T_{})".format(i)) for i in range(len(model.layers))]
    test_ixt_clusterd_bound = [tf.keras.metrics.Mean(name = r"Test I(X;T_{})_c".format(i)) for i in range(len(model.layers))]
    test_ity_clusterd_bound = [tf.keras.metrics.Mean(name = r"Test I(Y;T_{})_c".format(i)) for i in range(len(model.layers))]
    test_ixt_dual_bound = [tf.keras.metrics.Mean(name = r"Test I(X;T_{})_dual".format(i)) for i in range(len(model.layers))]
    test_ity_dual_bound = [tf.keras.metrics.Mean(name = r"Test I(Y;T_{})_dual".format(i)) for i in range(len(model.layers))]
    matrices = [train_loss, test_loss]
    matrices.extend(test_ixt_dual_bound)
    matrices.extend(test_ity_dual_bound)
    matrices.extend(test_ity_bound)
    matrices.extend(test_ity_clusterd_bound)
    matrices.extend(test_ixt_bound)
    matrices.extend(test_ixt_clusterd_bound)
    optimizer = tf.keras.optimizers.SGD(lr, beta)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    ind = 0
    num_of_clusters = [50, 50, 50]
    num_of_epochs_labels = 1000
    lr_labels = 5e-2
    for epoch in range(num_epochs):
        for batch in train_ds.take(batch_per_epoch):
            loss_value = train_step(batch, model , loss_fn, optimizer)
            if ind % num_iterations_to_print == 0:
                ixts, ixts_clusters = 0, 0
                for batch_test in  test_ds.take(1):
                    test_loss_val = test_step(model=model, batch=batch_test, loss_fn=loss_fn)

                    itys = get_ity_all_layers(model, batch_test, entropy_y = py.entropy(), num_of_epochs=num_of_epochs_labels, lr = lr_labels)

                    ixts = get_ixt_all_layers(model=model, x_test=batch_test[0], noisevar=noisevar)
                    ixts_clusters = get_ixt_all_layers_clusters(model=model, x_test=batch_test[0], num_of_clusters=num_of_clusters)
                    itys_clusters = get_ity_all_layers_clusters(model=model, x_test=batch_test[0], num_of_clusters=num_of_clusters, py_x=py_x, xs=xs, py=py)
                    information_dual_ib = get_information_dual_all_layers(model=model, num_of_clusters=num_of_clusters, xs=xs,
                                                    A=A, lambd=lambd, px=px, py=py)
                [test_ixt_dual_bound[i](information_dual_ib[i][0]) for i in range(len(information_dual_ib))]
                [test_ity_dual_bound[i](information_dual_ib[i][1]) for i in range(len(information_dual_ib))]
                [test_ixt_bound[i](ixts[i]) for i in range(len(ixts))]
                [test_ity_bound[i](itys[i]) for i in range(len(itys))]
                [test_ixt_clusterd_bound[i](ixts_clusters[i]) for i in range(len(ixts))]
                [test_ity_clusterd_bound[i](itys_clusters[i]) for i in range(len(itys_clusters))]
                train_loss(loss_value)
                test_loss(test_loss_val)
                log_summary(summary_writer, optimizer, epoch, matrices)
            ind += 1
def main(argv):
    if os.path.isdir(FLAGS.summary_path):
        shutil.rmtree(FLAGS.summary_path)

    # The widhts of the layers in the random network
    layer_widths = [10, 5]
    nonlin = FLAGS.nonlin
    train_ds, test_ds, py, py_x, xs, px, A, lambd= create_dataset(FLAGS.num_train, FLAGS.num_test,
                                       FLAGS.x_dim, layer_widths, nonlin,
                                       lambd_factor=FLAGS.lambd,
                                       alpha=FLAGS.alpha)
    model = bulid_model(FLAGS.input_shape, FLAGS.y_dim)
    train(model, train_ds, test_ds, FLAGS.lr, FLAGS.beta, FLAGS.num_epochs,
          FLAGS.batch_per_epoch,
          FLAGS.num_iterations_to_print, FLAGS.noisevar, py=py, py_x=py_x, xs=xs, px=px, A=A, lambd=lambd)
    logger.info('Done')


if __name__ == "__main__":

    app.run(main)