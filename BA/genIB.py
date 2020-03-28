from pylab import *
import numpy as np
from scipy.special import logsumexp, softmax
from scipy.stats import entropy


class GenIB:

    def __init__(self, dx, dy, pyx, dual=False, exp_dist=False, lambdas=None, A=None):
        self.idx = 0
        self.beta = np.asarray([0])
        self.dual = dual
        self.exp_dist = exp_dist
        self.k = dx
        self.dx = dx
        self.dy = dy
        self.dt = dx
        self.pyx = pyx
        # translate dists.
        self.px, self.py, self.py_x, self.px_y = self.translate_distributions()
        self.pt = np.asarray([self.px]).reshape((1, dx))
        self.py_t = np.asarray([self.py_x]).reshape((1, dy, dx))
        self.pt_x = np.zeros((1, dx, dx))
        self.px_t = np.zeros((1, dx, dx))
        if exp_dist:
            self.lambdas = lambdas
            self.A = A
            self.k, _ = A.shape
            self.py, self.py_x = self.calc_exp_data_dist()
            self.py_t = np.asarray([self.py_x]).reshape((1, dy, dx))
            self.At = np.asarray([self.A]).reshape((1, self.k, dx))
            self.lambdast = np.asarray([np.dot(self.py_t[0, :, :].T, self.lambdas)]).reshape((1, dx, self.k))
            logpy_t = -np.dot(self.lambdas, self.A)
            self.lambdast0 = logsumexp(logpy_t, axis=0).reshape((1, dx))
        return

    def update_state(self, beta):
        self.idx += 1
        self.beta = np.append(self.beta, beta)
        return

    def translate_distributions(self):
        px = np.sum(self.pyx, axis=0, keepdims=True)
        py = np.sum(self.pyx, axis=1, keepdims=True)
        px_y = (self.pyx / py).T
        py_x = self.pyx / px
        return px, py, py_x, px_y

    @staticmethod
    def dkl(p, q):
        """
        calc D_kl[p || q] = \sum_x p(x) * ln (p(x)/q(x))
        """
        _, dp = p.shape
        _, dq = q.shape
        dkl = np.zeros((dp, dq))
        for i in range(dp):
            for j in range(dq):
                dkl[i, j] = entropy(p[:, i], q[:, j])
        return dkl


    @staticmethod
    def fisher(p1, p2):
        _, dp = p1.shape
        F = np.zeros(dp)
        p_avg = 0.5 * (p1 + p2)
        for i in range(dp):
            F[i] = 0.5 * (entropy(p1[:, i], p_avg[:, i]) + entropy(p2[:, i], p_avg[:, i]))
        return F

    @staticmethod
    def calc_mi(px_y, px, py):
        """
           :param px_y: the conditional probability p(x|y) (|X|, |T|)
           :param py, px: The marginal probabilities
           :return: I_X_Y: The Mutual information I(X;Y) = <D_kl[p(x|y) || p(x)]>_{p(y)}
           """
        IXY = 0
        for i, py_i in enumerate(py):
            IXY += py_i * entropy(px_y[:, i], px)
        return IXY

    def calc_exp_data_dist(self):
        """
           calc p(y|x) = 1/Z e^(\sum_r lambda^(r)(y)A_r(x))
           :param lambdas: Lagrange multipliers, np.array((r, |y|))
           :param A: distribution moments, np.array((r, |x|))
           :return p_y_mid_x, p_y, A_0
           """
        logpy_x = -np.dot(self.lambdas, self.A)
        py_x = softmax(logpy_x, axis=0)
        py = np.dot(py_x, self.px.squeeze())
        return py, py_x

    def calc_dist(self, py_x_sample):
        px_t = self.px_t[self.idx, :, :]
        if self.exp_dist or self.dual:
            log_py_t = np.dot(np.log(py_x_sample), px_t)
            py_t = softmax(log_py_t, axis=0)
        else:
            py_t = np.dot(py_x_sample, px_t)
        return py_t

    def run_ba(self, beta):
        self.update_state(beta)
        py_t = self.py_t[self.idx - 1, :, :]
        pt_x = self.pt_x[self.idx - 1, :, :]
        pt = self.pt[self.idx - 1, :]
        py_x = self.py_x
        px = self.px
        if self.exp_dist:
            lambdast = self.lambdast[self.idx - 1, :, :]
            lambdast0 = self.lambdast0[self.idx - 1, :]
            At = self.At[self.idx - 1, :, :]
        i = 0
        while True:
            pt_x_prev = pt_x
            dkl_i = self.dkl(py_x, py_t).T if not self.dual else self.dkl(py_t, py_x)
            if self.exp_dist:
                dkl_i = np.zeros((self.dt, self.dx))
                for ti in range(self.dt):
                    lAt = np.dot(lambdast[ti, :], At[:, ti])
                    for xi in range(self.dx):
                        dkl_i[ti, xi] = np.dot(lambdast[ti, :], self.A[:, xi]) - lAt - lambdast0[ti]
            logpt_x = np.log(pt.reshape(self.dt, 1)) - (beta * dkl_i)
            pt_x = softmax(logpt_x, axis=0)
            pt = np.dot(pt_x, px.squeeze())
            px_t = (pt_x * px).T / pt.reshape(1, self.dt)
            if self.exp_dist:
                At = np.dot(self.A, px_t)
                logpy_t = -np.dot(self.lambdas, At)
                lambdast0 = logsumexp(logpy_t, axis=0)
                py_t = softmax(logpy_t, axis=0)
                lambdast = np.dot(py_t.T, self.lambdas)
            elif self.dual:
                log_py_t = np.dot(np.log(py_x), px_t)
                py_t = softmax(log_py_t, axis=0)
            else:
                py_t = np.dot(py_x, px_t)
            # stopping criteria
            diff_val = np.mean(self.fisher(pt_x, pt_x_prev)) if (i > 0) else np.inf
            if (diff_val < 1e-9) or (i > 1e+7):
                break
            i += 1
        self.pt_x = np.append(self.pt_x, pt_x.reshape((1, self.dx, self.dx)), axis=0)
        self.px_t = np.append(self.px_t, px_t.reshape((1, self.dx, self.dx)), axis=0)
        self.pt = np.append(self.pt, pt.reshape((1, self.dx)), axis=0)
        self.py_t = np.append(self.py_t, py_t.reshape((1, self.dy, self.dx)), axis=0)
        IXT = self.calc_mi(px_t, px.squeeze(), pt)
        if self.exp_dist:
            self.lambdast = np.append(self.lambdast, lambdast.reshape((1, self.dx, self.k)), axis=0)
            self.lambdast0 = np.append(self.lambdast0, lambdast0.reshape((1, self.dx)), axis=0)
            self.At = np.append(self.At, At.reshape((1, self.k, self.dx)), axis=0)
            py_t_markov = np.dot(py_x, px_t)
            py_markov = np.dot(py_t_markov, pt)
            IYT = self.calc_mi(py_t_markov, py_markov, pt)
        elif self.dual:
            py_t_markov = np.dot(py_x, px_t)
            py_markov = np.dot(py_t_markov, pt)
            IYT = self.calc_mi(py_t_markov, py_markov, pt)
        else:
            IYT = self.calc_mi(py_t, self.py.squeeze(), pt)
        return i, IXT, IYT

