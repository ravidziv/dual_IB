import numpy as np
import tensorflow as tf
from scipy.special import logsumexp, softmax
from sklearn.metrics import pairwise_distances
from MINE import MINE
from sklearn import mixture
from  tensorflow_probability import distributions as tfd


def GMM_entropy(dist, var, d, bound='upper', weights=None):
    # computes bounds for the entropy of a homoscedastic Gaussian mixture model [Kolchinsky, 2017]
    # dist: a matrix of pairwise distances
    # log_var: the log-variance of the mixture components
    # d: number of dimensions
    # n: number of mixture components
    n = dist.shape[0]

    if bound is 'upper':
        dist_norm = - dist / (2.0 * var)  # uses the KL distance
    elif bound is 'lower':
        dist_norm = - dist / (8.0 * var)  # uses the Bhattacharyya distance
    const = 0.5 * d * np.log(2.0 * np.pi * np.exp(1.0) * var) + np.log(n)
    if weights is not None:
        h = np.average(const, weights=weights) - np.average(logsumexp(dist_norm, 1, b=weights), weights=weights)
    else:
        h = np.average(const) - np.mean(logsumexp(dist_norm, 1))
    return h


def gaussian_entropy_np(d, var):
    # Entropy of a Gaussian distribution with 'd' dimensions and log variance 'log_var'
    h = 0.5 * d * (np.log(2.0 * np.pi * np.exp(1)) + np.log(var))
    return h



def get_information_MINE(xs, ts, batch_size=50, epochs=25, num_of_epochs_average=10):
    """return the information between x and t by the MINE estimator"""
    mine_model = MINE(x_dim=xs.shape[1], y_dim=ts.shape[1], batch_size=batch_size)
    #The zero loss is a workaround to remove the warning
    mine_model.compile(optimizer='adam', loss = {'output_1': lambda x,y : 0.0})
    fit_history = mine_model.fit(np.concatenate([xs, ts], axis=1), y=xs.numpy(), batch_size=batch_size, epochs=epochs,
                                 verbose=0)
    fit_loss = np.array(fit_history.history['loss'])
    return tf.reduce_mean(-fit_loss[-num_of_epochs_average:])


def get_ixt(inputs, noisevar=None):
    """Get I(X;T) assumeing that p(t|x) is a gsussian with noise variance (nonlinear IB)."""
    inputs = inputs.numpy().astype(np.float64)
    input_dim = inputs.shape[1]
    H_T_given_X = gaussian_entropy_np(input_dim, noisevar)
    dist_matrix = pairwise_distances(inputs)
    H_T_lb = GMM_entropy(dist_matrix, noisevar, input_dim, 'lower')
    Ixt_lb = H_T_lb - H_T_given_X
    H_T = GMM_entropy(dist_matrix, noisevar, input_dim, 'upper')
    Ixt = H_T - H_T_given_X  # nonlinear IB upper bound on I(X;T)
    return Ixt


def nce_estimator(num_of_samples, qxt):
    """The NCE estimator for I(X;T)"""
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


def mine_estimator(qxt, weights, means, num_of_samples=None):
    pts = tfd.Categorical(probs=weights)
    t = pts.sample(num_of_samples)
    ts = tf.stack([means[t[i]] for i in range(len(t))])
    xs = tf.stack([qxt[t[i]].sample() for i in range(len(t))])
    ixt = get_information_MINE(xs, ts, batch_size=500)
    return ixt


def get_ixt_clustered(data, num_of_clusters, num_of_samples):
    """Calculate ixt after clustring to mixture of gaussian"""
    clf = mixture.GaussianMixture(n_components=num_of_clusters, covariance_type='tied')
    clf.fit(data)
    qxt = tfd.MultivariateNormalTriL(loc=clf.means_, scale_tril=tf.linalg.cholesky(clf.covariances_))
    ixt_nce = nce_estimator(num_of_samples, qxt)
    ixt_mine = mine_estimator(qxt, clf.weights_, clf.means_, num_of_samples=num_of_samples)
    return ixt_nce, ixt_mine



def linear_estimator(py, log_py_ts):
    hyt = tf.reduce_mean(tf.cast(py._probs, tf.float64)[:, None] * tf.transpose(tf.stack(log_py_ts)))
    return tf.cast(py.entropy(), tf.float64) + hyt


def ity_mine_estimator(qyt, weights, means, num_of_samples):
    pts = tfd.Categorical(probs=weights)
    t = pts.sample(num_of_samples)
    ts = tf.stack([means[t[i]] for i in range(len(t))])
    xs = tf.stack([qyt[t[i]].sample() for i in range(len(t))])
    ixt = get_information_MINE(ts, xs[:, None], batch_size=500)
    return ixt


def get_iyt_clustered(data, num_of_clusters, py_x, py, num_of_samples):
    """Calculate ixt after clustring to mixture of gaussian"""
    clf = mixture.GaussianMixture(n_components=num_of_clusters, covariance_type='tied')
    clf.fit(data)
    qxt = tfd.MultivariateNormalTriL(loc=clf.means_, scale_tril=tf.linalg.cholesky(clf.covariances_))
    log_py_ts = []
    qyt = []
    for t_index in range(num_of_clusters):
        px_t = qxt[t_index].prob(tf.cast(data, tf.float64))
        py_t = tf.reduce_sum(tf.cast(py_x, tf.float64) * tf.cast(px_t[:, None], tf.float64), axis=0)
        py_t /= np.sum(py_t)
        log_py_ts.append(np.log(py_t))
        qyt.append(tfd.Categorical(probs=py_t))
    ity_linear = linear_estimator(py, log_py_ts)
    ity_mine = ity_mine_estimator(qyt, clf.weights_, clf.means_, num_of_samples=num_of_samples)
    return ity_linear, ity_mine


def get_information_all_layers_clusterd(model, x_test, num_of_clusters=None, num_of_samples=None, xs=None,
                                        py_x=None, py=None):
    """The infomration I(X;T) and I(T;Y) after you clustered T to mixture of gaussians"""
    ixts = [get_ixt_clustered(tf.keras.Model(model.inputs, model.layers[layer_indec].output)(x_test),
                              num_of_clusters=num_of_clusters[layer_indec], num_of_samples=num_of_samples)
            for layer_indec in range(len(model.layers))]
    itys = [get_iyt_clustered(tf.keras.Model(model.inputs, model.layers[layer_indec].output)(xs),
                              num_of_clusters=num_of_clusters[layer_indec], py_x=py_x, py=py, num_of_samples=num_of_samples)
            for layer_indec in range(len(model.layers))]
    return ixts, itys


def get_ixt_all_layers(model, x_test, noisevar):
    # Get I(X;T) for each layer
    ixts = [get_ixt(tf.keras.Model(model.inputs, layer.output)(x_test), noisevar=noisevar) for layer in model.layers]
    return ixts


def get_information_all_layers_MINE(model, x_test, y_test):
    # Get I(X;T) I(Y;T) for each layer with the MINE estimator
    ixts = [get_information_MINE(x_test, tf.keras.Model(model.inputs, layer.output)(x_test)) for layer in model.layers]
    iyts = [get_information_MINE(y_test, tf.keras.Model(model.inputs, layer.output)(x_test)) for layer in model.layers]
    return ixts, iyts
