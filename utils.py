import numpy as np
from scipy.special import logsumexp, softmax

def entropy(ps):
    return -np.sum([p*np.log(p) for p in ps if not np.isclose(p,0)])

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