import numpy as np
import tensorflow as tf
#from sklearn.metrics import pairwise_distances

from tensorflow_addons.losses.metric_learning import pairwise_distance as pairwise_distances
from estimators.MINE import get_information_MINE
from sklearn import mixture
from  tensorflow_probability import distributions as tfd
from network_utils import bulid_model
from estimators.dual_ib import calc_IXT_p, calc_IYT_p
import time
import math as m
pi = tf.constant(m.pi)


@tf.function
def GMM_entropy(dist, var, d, bound='upper', weights=None, n=1):
    # computes bounds for the entropy of a homoscedastic Gaussian mixture model [Kolchinsky, 2017]
    # dist: a matrix of pairwise distances
    # log_var: the log-variance of the mixture components
    # d: number of dimensions
    # n: number of mixture components
    if bound is 'upper':
        dist_norm = - dist / (2.0 * var)  # uses the KL distance
    elif bound is 'lower':
        dist_norm = - dist / (8.0 * var)  # uses the Bhattacharyya distance
    const = 0.5 * d * tf.math.log(2.0 * pi * tf.cast(tf.math.exp(1.0), tf.float32) * var) + tf.math.log(tf.cast(n, tf.float32))
    h = tf.reduce_mean(const) - tf.reduce_mean(tf.reduce_logsumexp(dist_norm, axis=1))
    return h

@tf.function
def gaussian_entropy_np(d, var):
    # Entropy of a Gaussian distribution with 'd' dimensions and log variance 'log_var'
    h = 0.5 * d * (tf.math.log(2.0 * pi * tf.cast(np.exp(1), tf.float32)) + tf.math.log(var))
    return h



@tf.function
def get_ixt_gmm(dist_matrix, input_dim = 0, noisevar=None):
    """Get I(X;T) assumeing that p(t|x) is a gsussian with noise variance (nonlinear IB)."""
    H_T_given_X = gaussian_entropy_np(input_dim, noisevar)
    #dist_matrix = pairwise_distances(tf.cast(inputs, tf.float32))
    #H_T_lb = GMM_entropy(dist_matrix, noisevar, input_dim, 'lower')
    #Ixt_lb = H_T_lb - H_T_given_X
    H_T = GMM_entropy(dist_matrix, noisevar, input_dim, 'upper', n=input_dim)
    Ixt = H_T - H_T_given_X  # nonlinear IB upper bound on I(X;T)
    return tf.cast(Ixt, tf.float64)


def nce_estimator(num_of_samples, qxt, sample_x, lower = False):
    """The NCE estimator for I(X;T) van den Oord et al. (2018):"""
    ixt_array = 0
    for i in range(num_of_samples):
        sample_x = qxt.sample()
        log_pxt = qxt.log_prob(tf.cast(sample_x[:, None], tf.float64))
        diag_xt = tf.math.diag(log_pxt)
        if lower:
            log_pxt_m_d = log_pxt[~tf.eye(log_pxt.shape[0], dtype=bool)].reshape(log_pxt.shape[0], -1)
        else:
            log_pxt_m_d = log_pxt
        K = log_pxt_m_d.shape[1]
        h_negative_yz = tf.reduce_logsumexp(log_pxt_m_d, axis=1) - tf.math.log(tf.cast(K, tf.float64))
        ixt = diag_xt - h_negative_yz
        ixt_array += tf.reduce_mean(ixt)
    ixt_array /= num_of_samples
    return ixt_array


def mine_estimator(qxt, pts, means, num_of_samples, epochs=100):
    """Returns I(X;T) based on the MINE estimtor."""
    t = pts.sample(num_of_samples)
    ts = tf.stack([means[t[i]] for i in range(len(t))])
    xs = tf.stack([qxt[t[i]].sample() for i in range(len(t))])
    ixt = get_information_MINE(xs, ts, batch_size=500, epochs=epochs)
    return ixt


@tf.function
def get_ixt_clustered(dist_matrix, input_dim=0, variance = 1):
    """Calculate ixt after clustering to mixture of gaussian."""
    #ixt_nce = nce_estimator(num_of_samples, qxt, sample_x=xs)
    #ixt_mine = get_information_MINE(xs, ts, batch_size=500, epochs=mine_epochs)
    #ixt_mine = ixt_nce
    ixt_nonlinear =  get_ixt_gmm(dist_matrix, input_dim=input_dim, noisevar=tf.cast(variance, tf.float32))
    return ixt_nonlinear


def bayes_estimator(py_entropy, py_ts_entropy, pts_prob):
    """The optimal bayes estimator."""
    tensor  =pts_prob*py_ts_entropy
    tensor_without_nans = tf.where(tf.math.is_nan(tensor), tf.zeros_like(tensor), tensor)
    hyt =tf.reduce_sum(tensor_without_nans)
    return tf.cast(py_entropy, tf.float64) - tf.cast(hyt, tf.float64)


@tf.function
def get_iyt_clustered(py_ts_entropy, py_entropy, pts_prob):
    """Calculate ixt after clustring to mixture of gaussian"""
    ity_linear = bayes_estimator(py_entropy=py_entropy, py_ts_entropy=py_ts_entropy , pts_prob=pts_prob)
    #ity_mine = ity_linear
    #ity_mine = mine_estimator(qyt, pts, means, num_of_samples=num_of_samples, m)
    return ity_linear


@tf.function
def get_probs(data, py_x, px_probs, cov, means):
    """Calculate probs of x|t, t and y|t based on the data and the meand and the variances of the clusters."""
    with tf.device('/CPU:0'):
        data_i = tf.cast(data, tf.float64)
        qt_x = tfd.MultivariateNormalDiag(loc=tf.cast(means, tf.float64),
                                          scale_diag=tf.transpose(cov))
        pts = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(
                probs=tf.cast(px_probs, tf.float64)),
            components_distribution=qt_x)
        # Bayes rule
        pts_prob = tf.cast(pts.prob(data_i), tf.float64)
        pt_x_prob = qt_x.prob(data_i[:, None, :])
        pts_prob = tf.math.divide_no_nan(pts_prob, tf.reduce_sum(pts_prob, axis=0))
        pt_x_prob = tf.math.divide_no_nan(pt_x_prob, tf.reduce_sum(pt_x_prob, axis=0))
        px_t = tf.math.divide_no_nan(tf.transpose((pt_x_prob * tf.cast(px_probs, tf.float64)[None, :])),  pts_prob[:, None])
        px_t = tf.math.divide_no_nan(px_t, tf.reduce_sum(px_t, axis=0))
        py_t = tf.einsum('ki,kj->ij', py_x, px_t)
    return tf.cast(px_t, tf.float64) , tf.cast(pts_prob, tf.float64), py_t[0, :]

def custered_mixture(data, num_of_cluster):
    """Calculates the GMM of the data."""
    clf = mixture.GaussianMixture(n_components=num_of_cluster, covariance_type='spherical', reg_covar=1e-4)
    clf.fit(data)
    ts = clf.predict(data)
    return clf.means_, clf.covariances_, ts, clf.means_[ts], clf.covariances_[ts]

@tf.function
def cluster_data(data, num_of_cluster, px, py_x, max_num = 5000 ):
    """Cluster the data of the given layer and calculate all its probs"""
    #TODO- decide test/train data
    means_, covariances_, ts, means_ts, covariances_ts =tf.numpy_function(custered_mixture, [data, num_of_cluster], [tf.float64, tf.float64, tf.int64, tf.float64, tf.float64])
    cov = tf.cast(covariances_ts * tf.ones((data.shape[1], len(covariances_ts)), dtype=tf.float64), tf.float64)
    qt_x = tfd.MultivariateNormalDiag(loc=tf.cast(means_ts, tf.float64),
                                      scale_diag=tf.transpose(cov))

    px_t, pts_prob,py_t  = get_probs(data, py_x, px.probs, cov, means_ts)
    qyt = tfd.Bernoulli(probs=py_t)
    with tf.device('/GPU:0'):
        dist_matrix = pairwise_distances(tf.cast(qt_x.mean(), tf.float32))
    return qyt.entropy(), px_t , dist_matrix, pts_prob,ts, means_, covariances_


@tf.function
def get_information(A, lambd, betas, calc_dual, px_entropy, py_entropy, dists, qyt_entropy,
                    pts_prob, px_t, input_dim, size, dual_inforamtion_arr=0, inforamtion_arr=0):
    inforamtion_arr = tf.TensorArray(tf.float64, size=size)
    dual_inforamtion_arr = tf.TensorArray(tf.float64, size=size)
    for i in range(size):
        beta_s = betas.read(i)
        dist_matrix =dists.read( i)
        qyt_entropy_c = qyt_entropy.read(i)
        pts_prob_c = pts_prob.read(i)
        px_t_c = px_t.read(i)
        ixt = get_ixt_clustered(dist_matrix=dist_matrix, input_dim=input_dim, variance=beta_s)
        ity = get_iyt_clustered(py_ts_entropy=qyt_entropy_c, py_entropy=py_entropy,
                                pts_prob=pts_prob_c)
        inforamtion_arr =inforamtion_arr.write(i,tf.cast( tf.stack([ixt, ity], axis=0), tf.float64))
        if calc_dual:
            ixt_dual = calc_IXT_p(pts_prob_c, px_t_c, A, lambd, beta_s, px_entropy)
            ity_dual = calc_IYT_p(pts_prob_c, px_t_c, A, lambd, py_entropy)
            dual_inforamtion_arr=dual_inforamtion_arr.write(i,tf.cast(tf.stack([ixt_dual, ity_dual]), tf.float64))
    return inforamtion_arr.stack(), dual_inforamtion_arr.stack()


@tf.function
def get_information_all_layers_clusterd(data_all, num_of_clusters=None,
                                        py_x=None, py=None, px=None, calc_dual=True, A=None,
                                        lambd=None):

    """The infomration I(X;T) and I(T;Y) after you clustered T to mixture of gaussians"""
    py_entropy = py.entropy()
    px_entropy = px.entropy()
    betas = tf.TensorArray(tf.float64, size=len(num_of_clusters)*len(num_of_clusters[0]))
    dists = tf.TensorArray(tf.float32, size=len(num_of_clusters)*len(num_of_clusters[0]))
    qyt_entropy = tf.TensorArray(tf.float64, size=len(num_of_clusters)*len(num_of_clusters[0]))
    pts_prob = tf.TensorArray(tf.float64, size=len(num_of_clusters)*len(num_of_clusters[0]))
    px_t = tf.TensorArray(tf.float64, size=len(num_of_clusters)*len(num_of_clusters[0]))

    for layer_index , (data, layer_num_clusters) in enumerate(zip(data_all, num_of_clusters)):
        t =tf.timestamp()
        for i in tf.range(len(layer_num_clusters)):
            ind = tf.cast(layer_index*len(num_of_clusters[0]), tf.int32)+tf.cast(i, tf.int32)
            if ind % 10 ==0:
                tf.print ( ind, i, len(layer_num_clusters), tf.timestamp() - t)
            t = tf.timestamp()
            try:
                qyt_entropy_c, px_t_c, dist_matrix, pts_prob_c, ts, means, convarinaces = cluster_data(data, layer_num_clusters[i], px, py_x )
            except:
                pass
            beta_s = tf.cast(tf.reduce_mean(convarinaces), tf.float64)
            betas=betas.write(ind, beta_s)
            dists=dists.write(ind, dist_matrix)
            qyt_entropy=qyt_entropy.write(ind, qyt_entropy_c)
            pts_prob=pts_prob.write(ind, pts_prob_c)
            px_t=px_t.write(ind, px_t_c)
    inforamtion_arr, dual_inforamtion_arr = get_information(A, lambd, betas, calc_dual, px_entropy, py_entropy, dists, qyt_entropy,
                    pts_prob, px_t, input_dim=data_all[0].shape[0],size=len(num_of_clusters)*len(num_of_clusters[0]))
    return inforamtion_arr, dual_inforamtion_arr



def get_nonlinear_information(model, batch_test, entropy_y, num_of_epochs_inf_labels, lr_labels, noisevar):
    information = []
    for layer_index in range(len(model.layers)):
        x_test, targets = batch_test
        pred = tf.keras.Model(model.inputs, model.layers[layer_index].output)(x_test)
        ity = get_iyt_from_top_model([pred, targets],
                                           entropy_y=entropy_y,
                                           num_of_epochs=num_of_epochs_inf_labels, lr=lr_labels)
        ixt = get_ixt_gmm(pred, noisevar=noisevar)
        information.append(ixt, ity)
    return information

def get_iyt_from_top_model(batch, entropy_y, num_of_epochs, lr=1e-3, layers_width=[10, 10]):
    """Train network to fit logp(y|t)."""
    pred, targets = batch
    model = bulid_model(pred.shape[1], targets.shape[-1], layers_width=layers_width)
    optimizer = tf.keras.optimizers.Adam(lr)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=['accuracy'])
    model.fit(x = batch[0], y = batch[1], epochs = num_of_epochs, verbose =0)
    loss_value = model.evaluate(x=batch[0], y= batch[1], verbose=0)
    return entropy_y- loss_value

