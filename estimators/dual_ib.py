"""Functions for calculating the dual IB.


"""
import numpy as np
from scipy.special import logsumexp, softmax
import tensorflow as tf
from sklearn import mixture
from  tensorflow_probability import distributions as tfd


def calc_At(px_t, A):
  return np.dot(A, px_t)

def calc_lambdt(py_t, lambd):
  return np.dot(lambd, py_t)

def calc_py_t(At, lambd):
  '''
  returns: p(y|t) and \log(Z)
  '''
  log_py_t = -np.dot(lambd.T, At)
  lambdt0 = logsumexp(log_py_t, axis=0)
  return softmax(log_py_t, axis=0), lambdt0

def calc_d_dualib(A, At, lambd, lambd0, lambdt, lambdt0):
  return lambd0-lambdt0+np.dot(lambdt.T, (A-At))

def calc_pt_x(pt, A, At, lambd, lambdt, lambdt0, beta):
    Am = A[:, :, None]-At[:, None, :]
    res = np.einsum('kij,kj->ij',Am, lambdt)
    log_pt_x = beta*(lambdt0-res) + np.log(pt)
    logZt_x = logsumexp(log_pt_x, axis=0)
    return softmax(log_pt_x, axis=0), logZt_x

# info calc with px_t:
def calc_IYT_p(pt, px_t, A, lambd, HY):
  At = calc_At(px_t, A)
  py_t, lambdt0 = calc_py_t(At, lambd)
  lambdt = calc_lambdt(py_t, lambd)
  At_lambd  =np.einsum('ij,ij->j',At, lambdt)
  HY_T = np.dot(pt, At_lambd+lambdt0)
  return HY - HY_T

def calc_IXT_p(pt, px_t, A, lambd, beta, HX):
  At = calc_At(px_t, A)
  py_t, lambdt0 = calc_py_t(At, lambd)
  lambdt = calc_lambdt(py_t, lambd)
  _, logZt_x = calc_pt_x(pt, A, At, lambd, lambdt, lambdt0, beta)
  HX_T = -beta*np.dot(pt, lambdt0) + np.mean(logZt_x)
  return HX - HX_T

def get_information_dual(data, num_of_clusters, A, lambd, x_entopy, y_entopy, beta_func):
    """Calculate ixt after clustring to mixture of gaussian"""
    clf = mixture.GaussianMixture(n_components=num_of_clusters, covariance_type='tied')
    clf.fit(data)
    qxt = tfd.MultivariateNormalTriL(loc=clf.means_, scale_tril=tf.linalg.cholesky(clf.covariances_))
    pt = clf.weights_
    beta = beta_func(clf.covariances_)

    px_t = []
    for t_index in range(num_of_clusters):
        px_t.append(qxt[t_index].prob(tf.cast(data, tf.float64)))
    px_t = tf.transpose(tf.stack(px_t))
    ixt = calc_IXT_p(pt, px_t.numpy(), A.numpy().T, lambd.numpy().T, beta, x_entopy)
    ity = calc_IYT_p(pt, px_t.numpy(), A.numpy().T, lambd.numpy().T, y_entopy)
    return ixt, ity


def get_information_dual_all_layers(model, num_of_clusters=None, xs=None,
                                    A=None, lambd=None, px=None, py=None, beta_func=None):
    information = [get_information_dual(tf.keras.Model(model.inputs, model.layers[layer_indec].output)(xs),
                                 num_of_clusters=num_of_clusters[layer_indec], A=A, lambd=lambd, x_entopy=px.entropy(),
                                 y_entopy=py.entropy(), beta_func=beta_func) for layer_indec in range(len(model.layers))]
    return information



def beta_func(cov):
    """Return beta from the covarinces of the clusters."""
    return 1/ np.sqrt(np.trace(cov))
