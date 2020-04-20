import numpy as np
import tensorflow as tf
from estimators.information_utils import calc_information_from_mat, extract_probs
from functools import partial

def get_unique_probs(x):
    uniqueids = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    _, unique_inverse, unique_counts = np.unique(uniqueids, return_index=False, return_inverse=True, return_counts=True)
    return np.asarray(unique_counts / float(sum(unique_counts))), unique_inverse

def bin_information(layerdata, labelixs, binsize):
    def get_entropy(d):
        digitized = np.floor(d / binsize).astype('int')
        p_ts, _ = get_unique_probs(digitized)
        return -np.sum(p_ts * np.log2(p_ts))

    H_LAYER = get_entropy(layerdata)
    H_LAYER_GIVEN_OUTPUT = 0
    for label, ixs in labelixs.items():
        H_LAYER_GIVEN_OUTPUT += ixs.mean() * get_entropy(layerdata[ixs, :])
    return H_LAYER, H_LAYER - H_LAYER_GIVEN_OUTPUT

def get_information_bins_estimators2(batch_test, ts, binsize=[.5]):
    """Calculate the information for the network for all the epochs and all the layers"""
    x_test, labels = batch_test
    #bins = np.linspace(-1, 1, num_of_bins)
    pxs, pys, unique_inverse_x, unique_inverse_y = extract_probs(labels.numpy(), x_test.numpy())
    partial_func = partial(information_bins2,  pys=pys, pxs = pxs, unique_inverse_x=unique_inverse_x,
                           py_x=labels.numpy().T)
    information = []
    for layer_index in range(len(ts)):
        for bin_index in range(len(binsize)):
            current_information = partial_func(ts[layer_index].numpy(), binsize=binsize[bin_index])
            information.append(current_information)
    return information


def information_bins2(data, binsize, pys, pxs, unique_inverse_x, py_x):
    #digitized = bins[np.digitize(np.squeeze(data.reshape(1, -1)), bins) - 1].reshape(len(data), -1)
    digitized = np.floor(data / binsize).astype('int')
    b2 = np.ascontiguousarray(digitized).view(
		np.dtype((np.void, digitized.dtype.itemsize * digitized.shape[1])))
    _, unique_inverse_t, unique_index, unique_counts = \
		np.unique(b2, return_index=True, return_inverse=True, return_counts=True)
    px_t = np.zeros((pxs.shape[0],unique_inverse_t.shape[0]) )
    for t_index in range(len(unique_counts)):
        xs_ts = unique_index == t_index
        px_t[xs_ts,t_index] =1
    p_ts = unique_counts / float(sum(unique_counts))
    px_t /= np.sum(px_t, axis=0)
    py_t = np.einsum('ij,jt->it', py_x, px_t)
    ity, ixt =calc_information_from_mat(pys, p_ts, py_t)
    return ixt, ity

ent = lambda pt: -tf.reduce_sum(pt * tf.math.log.log(pt), axis=0)

@tf.function
def calc_information_from_mat_tf(py, pt, py_t):
    """Calculate the MI based on binning of the data"""
    Ht = ent(pt)
    hy = ent(py)
    HYT =tf.reduce_sum(pt*(ent(py_t)))
    #H2X = calc_condtion_entropy(px, data, unique_inverse_x)
    IY = hy - HYT
    return IY, Ht

