"""Functions for calculating the dual IB.


"""
import tensorflow as tf
from sklearn import mixture
from  tensorflow_probability import distributions as tfd


def calc_At(px_t, A):
  return tf.tensordot(A, px_t, axes=1)

def calc_lambdt(py_t, lambd):
  return tf.tensordot(lambd, py_t, axes=1)

def calc_py_t(At, lambd):
  '''
  returns: p(y|t) and \log(Z)
  '''
  log_py_t = -tf.einsum('ij,ik->jk', lambd, At)
  lambdt0 = tf.reduce_logsumexp(log_py_t, axis=0)
  ex = tf.exp(log_py_t)
  sum_s = tf.reduce_sum(ex, axis=0)
  return ex/sum_s, lambdt0

def calc_d_dualib(A, At, lambd, lambd0, lambdt, lambdt0):
  return lambd0-lambdt0+tf.tensordot(tf.transpose(lambdt), (A-At), axes=1)

def calc_pt_x(pt, A, At, lambd, lambdt, lambdt0, beta):
    Am = A[:, :, None]-At[:, None, :]
    res = tf.einsum('kij,kj->ij',Am, lambdt)
    log_pt_x = beta*(lambdt0-res) + tf.math.log(pt+1e-20)
    log_pt_x = tf.where(tf.math.is_nan(log_pt_x), tf.ones_like(log_pt_x) * 1, log_pt_x)
    logZt_x = tf.reduce_logsumexp(log_pt_x, axis=0)
    ex = tf.exp(log_pt_x)
    sum_s = tf.reduce_sum(ex, axis=0)
    return ex/sum_s, logZt_x

#@tf.function
def calc_IYT_p(pt, px_t, A, lambd, HY):
  At = calc_At(px_t, A)
  py_t, lambdt0 = calc_py_t(At, lambd)
  lambdt = calc_lambdt(py_t, lambd)
  At_lambd  =tf.einsum('ij,ij->j',At, lambdt)
  HY_T = tf.tensordot(pt, At_lambd+lambdt0, axes=1)
  return tf.cast(HY - HY_T, tf.float64)

#@tf.function
def calc_IXT_p(pt, px_t, A, lambd, beta, HX):
  At = calc_At(px_t, A)
  py_t, lambdt0 = calc_py_t(At, lambd)
  lambdt = calc_lambdt(py_t, lambd)
  _, logZt_x = calc_pt_x(pt, A, At, lambd, lambdt, lambdt0, beta)
  HX_T = -beta* tf.reduce_sum(pt*lambdt0) + tf.reduce_mean(logZt_x)
  return tf.cast(tf.cast(HX, tf.float64) - HX_T, tf.float64)


def beta_func(cov):
    """Return beta from the covarinces of the clusters."""
    return 1/ tf.sqrt(tf.math.trace(cov))
