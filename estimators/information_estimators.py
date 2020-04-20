import numpy as np
import tensorflow as tf
#from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from tensorflow_addons.losses.metric_learning import pairwise_distance as pairwise_distances
from estimators.MINE import train_mine
from estimators.information_utils import calc_information_from_mat
from sklearn import mixture
from  tensorflow_probability import distributions as tfd
from network_utils import bulid_model
from estimators.dual_ib import calc_IXT_p, calc_IYT_p
import time
from functools import  partial
from estimators.kde import get_ixt_gmm

ent = lambda pt: -tf.reduce_sum(pt * tf.math.log(pt), axis=0)

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
    ixt = train_mine(xs, ts, batch_size=500, epochs=epochs)
    return ixt


@tf.function
def get_ixt_clustered(dist_matrix, input_dim=0, n=1, variance = 1):
    """Calculate ixt after clustering to mixture of gaussian."""
    #ixt_nce = nce_estimator(num_of_samples, qxt, sample_x=xs)
    #ixt_mine = get_information_MINE(xs, ts, batch_size=500, epochs=mine_epochs)
    #ixt_mine = ixt_nce
    ixt_nonlinear, Ht =  get_ixt_gmm(dist_matrix, input_dim=input_dim, n =n , noisevar=tf.cast(variance, tf.float32))
    return ixt_nonlinear, Ht

@tf.function
def bayes_estimator(py_entropy, py_ts_entropy, pts_prob):
    """The optimal bayes estimator."""
    tensor  = pts_prob*py_ts_entropy
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
def get_probs(data, py_x, px_probs, py, cov, means):
    """Calculate probs of x|t, t and y|t based on the data and the meand and the variances of the clusters."""
    with tf.device('/CPU:0'):
        data_i = tf.cast(data, tf.float64)
        qt_x = tfd.MultivariateNormalDiag(loc=tf.cast(means, tf.float64),
                                          scale_diag=tf.transpose(cov))
        pt_x_prob = qt_x.prob(data_i[:, None, :]) +1e-100
        pt_x_prob = tf.math.divide_no_nan(pt_x_prob, tf.reduce_sum(pt_x_prob, axis=0))
        ptx = pt_x_prob *px_probs
        pts = tf.reduce_sum(ptx, axis=1)
        px_t = tf.transpose(tf.math.divide_no_nan(ptx,  pts[:, None]))
        #todo - check this!
        #px_t = tf.transpose(px_t)
        py_t = tf.einsum('ki,kj->ij', py_x, px_t)
        pyt = py_t*pts
        pt_y = tf.transpose(tf.math.divide_no_nan(pyt,  py[:, None]))
    return tf.cast(px_t, tf.float64) , tf.cast(pts, tf.float64), py_t, pt_y

#@tf.function
def cluster_data(clustered_data, data, px, py, py_x, max_num = 5000 ):
    """Cluster the data of the given layer and calculate all its probs"""
    #TODO- decide test/train data
    means_, covariances_, ts, means_ts, covariances_ts = clustered_data
    cov = tf.cast(covariances_ts * tf.ones((data.shape[1], len(covariances_ts)), dtype=tf.float64), tf.float64)
    #qt_x = tfd.MultivariateNormalDiag(loc=tf.cast(means_ts, tf.float64),
    #                                  scale_diag=tf.transpose(cov))

    px_t, pts_prob,py_t, pt_y  = get_probs(data, py_x, px.probs, py.probs, cov, means_ts)
    #qyt = tfd.Categorical(probs=tf.transpose(py_t))
    with tf.device('/GPU:0'):
        dist_matrix = pairwise_distances(tf.cast(tf.cast(means_ts, tf.float64), tf.float32))
    return ent(py_t), px_t , dist_matrix, pts_prob,ts, means_, covariances_, means_ts, covariances_ts


#@tf.function(experimental_relax_shapes=True)
def get_information(A, lambd, betas, calc_dual, px_entropy, py_entropy, dists, qyt_entropy,
                    pts_prob, px_t, input_dim, n, size, num_of_epochs_inf_labels=1000, targets=None, preds=None):
    inforamtion_arr = tf.TensorArray(tf.float64, size=size)
    dual_inforamtion_arr = tf.TensorArray(tf.float64, size=size)
    for i in range(size):
        beta_s = betas.read(i)
        dist_matrix =dists.read( i)
        qyt_entropy_c = qyt_entropy.read(i)
        pts_prob_c = pts_prob.read(i)
        px_t_c = px_t.read(i)
        input_dim_c = input_dim.read(i)
        ixt,Ht = get_ixt_clustered(dist_matrix=dist_matrix, input_dim=input_dim_c, n=n, variance=beta_s)
        ity = get_iyt_clustered(py_ts_entropy=qyt_entropy_c, py_entropy=py_entropy,
                                pts_prob=pts_prob_c)
        pred =  preds[i]
        ity1 = get_iyt_from_top_model(pred, targets, py_entropy, num_of_epochs_inf_labels, lr=1e-4, layers_width=[50, 30, 10])

        print('______________', i, ixt, ity1 )

        inforamtion_arr =inforamtion_arr.write(i,tf.cast( tf.stack([ixt, ity, ity1], axis=0), tf.float64))
        if calc_dual:
            ixt_dual = calc_IXT_p(pts_prob_c, px_t_c, A, lambd, 1/beta_s, px_entropy)
            ity_dual = calc_IYT_p(pts_prob_c, px_t_c, A, lambd, py_entropy)
            dual_inforamtion_arr=dual_inforamtion_arr.write(i,tf.cast(tf.stack([ixt_dual, ity_dual]), tf.float64))
        else:
            dual_inforamtion_arr=dual_inforamtion_arr.write(i,tf.cast(tf.stack([0.0, 0.0]), tf.float64))

    return inforamtion_arr.stack(), dual_inforamtion_arr.stack()



def cluster_mixture(clf, data, train_data):
    """Calculates the GMM of the data."""
    clf.fit(train_data)
    ts = clf.predict(data)
    return clf.means_, clf.covariances_, ts, clf.means_[ts], clf.covariances_[ts]


#@tf.function
def get_information_all_layers_clusterd(data_all, train_data_all, clustered_data, num_of_clusters=None,
                                        py_x=None, py=None, px=None, calc_dual=True, A=None,
                                        lambd=None, clfs=None, targets=None, num_of_epochs_inf_labels=2000):

    """The information I(X;T) and I(T;Y) after you clustered T to mixture of gaussians"""
    py_entropy = py.entropy()
    px_entropy = px.entropy()
    size = len(num_of_clusters)*len(num_of_clusters[0])
    inforamtion_arr = tf.TensorArray(tf.float64, size=size)
    dual_inforamtion_arr = tf.TensorArray(tf.float64, size=size)
    n = data_all[0].shape[0]
    for layer_index , (data, train_data, layer_num_clusters) in enumerate(zip(data_all, train_data_all, num_of_clusters)):
        input_dim_c =data.shape[1]
        for i in tf.range(len(layer_num_clusters)):
            clustered_data_c = cluster_mixture(clfs[layer_index][i], data, data)
            ind = tf.cast(layer_index*len(num_of_clusters[0]), tf.int32)+tf.cast(i, tf.int32)
            qyt_entropy_c, px_t_c, dist_matrix, pts_prob_c, ts, means, convarinaces, means_ts, covariances_ts = \
                cluster_data(clustered_data_c, data , px, py, py_x )

            beta_s = tf.cast(tf.reduce_mean(convarinaces), tf.float64)
            ixt, Ht = get_ixt_clustered(dist_matrix=dist_matrix, input_dim=input_dim_c, n=n, variance=beta_s)
            ity1 = 0
            ity = get_iyt_clustered(py_ts_entropy=qyt_entropy_c, py_entropy=py_entropy,
                                    pts_prob=pts_prob_c)
            ity1 = get_iyt_from_top_model(means_ts, targets, py_entropy, num_of_epochs_inf_labels, lr=1e-4,
                                          layers_width=[30, 20, 10])
            #print('______________', ind, ixt, ity1)
            inforamtion_arr = inforamtion_arr.write(ind, tf.cast(tf.stack([ixt, ity, ity1], axis=0), tf.float64))
            if calc_dual:
                ixt_dual = calc_IXT_p(pts_prob_c, px_t_c, A, lambd, 1 / beta_s, px_entropy)
                ity_dual = calc_IYT_p(pts_prob_c, px_t_c, A, lambd, py_entropy)
                dual_inforamtion_arr = dual_inforamtion_arr.write(ind,
                                                                  tf.cast(tf.stack([ixt_dual, ity_dual]), tf.float64))
            else:
                dual_inforamtion_arr = dual_inforamtion_arr.write(ind, tf.cast(tf.stack([0.0, 0.0]), tf.float64))


    #inforamtion_arr, dual_inforamtion_arr = get_information(A, lambd, betas, calc_dual, px_entropy, py_entropy, dists, qyt_entropy,
    #                pts_prob, px_t, input_dim=input_dim,n=data_all[0].shape[0],
    #                                                        size=len(num_of_clusters)*len(num_of_clusters[0]),
    #                                                        targets=targets, preds=preds)
    return inforamtion_arr.stack(), dual_inforamtion_arr.stack()


@tf.function
def get_probs_all(py_x, pt_x_prob, px_probs, py_probs):
    ptx = pt_x_prob * px_probs
    pts = tf.reduce_sum(ptx, axis=1)
    px_t = tf.transpose(tf.math.divide_no_nan(ptx, pts[:, None]))
    py_t = tf.einsum('ki,kj->ij', py_x, px_t)
    pyt = py_t * pts
    pt_y = tf.transpose(tf.math.divide_no_nan(pyt, py_probs[:, None]))
    return pt_y, pts, py_t


@tf.function
def get_gaussian_infromation(pred, px_probs, py_probs, dist_matrix, py_x, noise_var):
    qt_x = tfd.MultivariateNormalDiag(loc=tf.cast(pred, tf.float64),
                                      scale_identity_multiplier=tf.cast(
                                          tf.ones((pred.shape[0],)) * noise_var,
                                          tf.float64))
    pt_x_prob = qt_x.prob(tf.cast(pred[:, None, :], tf.float64)) + 1e-40

    pt_x_prob = tf.math.divide_no_nan(pt_x_prob, tf.reduce_sum(pt_x_prob, axis=0))
    qt_y, pts, py_t = get_probs_all(py_x, pt_x_prob, px_probs, py_probs)
    #pt_y = tfd.Categorical(probs=tf.transpose(tf.cast(qt_y, tf.float64)))
    ixt_gmm_part = partial(get_ixt_gmm, dist_matrix=dist_matrix,
                           input_dim=tf.constant(pred.shape[1]),
                           n=tf.constant(pred.shape[0]))
    ixt, H_T = ixt_gmm_part(noisevar=noise_var)

    H_T_given_Y = tf.cast(tf.reduce_sum(py_probs * ent(qt_y)), tf.float64)
    #pts_s = tfd.Categorical(probs=tf.transpose(tf.cast(pts, tf.float64)))
    ity = ent(pts) - H_T_given_Y
    #iyt1, ixt1 = calc_information_from_mat(py_probs, pts, py_t)
    return ixt, ity


def get_nonlinear_information(ts, batch_test, entropy_y, num_of_epochs_inf_labels, lr_labels, noisevar, py_probs,
                              py_x, px_probs, num_of_samples =1, paths='', layer_width =[30, 20, 10]):
    x_test, targets = batch_test

    information = tf.TensorArray(tf.float64, size=len(ts)*len(noisevar))
    normalized_information =  tf.TensorArray(tf.float64, size=len(ts)*len(noisevar))
    new_targets = tf.expand_dims(targets, 0)
    new_targets = tf.repeat(new_targets, num_of_samples, 0)
    new_targets = tf.reshape(new_targets, (-1, targets.shape[1]))
    t_be = time.time()
    for layer_index in range(len(ts)):
        print (time.time()-t_be)
        t_be = time.time()

        pred = ts[layer_index]
        dist_matrix = pairwise_distances(tf.cast(pred, tf.float32))
        #ity = get_iyt_from_top_model(pred, targets, entropy_y, num_of_epochs_inf_labels, lr=5e-4, layers_width=[50, 20])
        #ity_adaptive = ity
        t = time.time()
        for j in tf.range(len(noisevar)):
            noise_var_inner = tf.gather(noisevar, j)
            qt_x = tfd.MultivariateNormalDiag(loc=tf.cast(pred, tf.float32),
                                              scale_identity_multiplier=noise_var_inner)
            t=time.time()
            pred_gaussian = qt_x.sample(num_of_samples)
            pred_gaussian = tf.reshape(pred_gaussian, (-1, pred.shape[1]))

            ind = tf.cast(layer_index * len(noisevar), tf.int32) + tf.cast(j, tf.int32)

            ity = get_iyt_from_top_model(pred_gaussian, new_targets, entropy_y, num_of_epochs_inf_labels, lr=lr_labels,
                                         layers_width=layer_width, path = paths[layer_index][j])
            t = time.time()
            ixt, _ = get_gaussian_infromation(pred, px_probs, py_probs, dist_matrix, py_x, noise_var_inner)
            t = time.time()
            ixt_adaptive, ity_adaptive = 0,0
            if False:
                ixt_adaptive, _ =\
                get_gaussian_infromation(pred, px_probs, py_probs, dist_matrix, py_x, np.max(pred)*noise_var_inner)
                ity_adaptive = get_iyt_from_top_model(pred_gaussian, new_targets, entropy_y, num_of_epochs_inf_labels, lr=5e-4,
                                             layers_width=[30, 20])

            a=information.write(ind,tf.cast(tf.stack([ixt, ity]), tf.float64))
            a.mark_used()
            a=normalized_information.write(ind,tf.cast(tf.stack([ixt_adaptive, ity_adaptive]), tf.float64))
            a.mark_used()
    return information.stack(), normalized_information.stack()


def get_iyt_from_top_model(pred, targets, entropy_y, num_of_epochs, lr=1e-3, layers_width=[10, 10], path=None):
    """Train network to fit logp(y|t)."""
    es_callbatck = tf.keras.callbacks.EarlyStopping(
        monitor='loss', min_delta=1e-6, patience=10, verbose=0, mode='auto',
        baseline=None, restore_best_weights=True
    )
    model = bulid_model(pred.shape[1], targets.shape[-1], layers_width=layers_width)
    #if path is not None:
    #    model.load_weights(path)
    optimizer = tf.keras.optimizers.Adam(lr)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=['accuracy'])
    model.fit(x = pred, y = targets, epochs = num_of_epochs, verbose =0, bath_size = 64, callbacks=[es_callbatck])
    loss_value = model.evaluate(x=pred, y= targets, verbose=0)
    return tf.cast(entropy_y- loss_value[0], tf.float64)