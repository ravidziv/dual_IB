import numpy as np
import BA.genIB as genIB
from scipy.stats import entropy


def my_run():
    # set up problem
    dy = 2
    dx = 5
    dt = dx

    x = np.linspace(-1, .6, dx)
    y = np.asarray([-1, 1])
    A = np.asarray([x]).reshape(1, dx)

    lambdas = np.asarray([y]).reshape(dy, 1)
    px = np.ones(dx) / dx

    logpy_x = -np.dot(lambdas, A)
    py_x = genIB.softmax(logpy_x, axis=0)
    py = np.dot(py_x, px.squeeze())
    pyx = py_x * px
    px_y = (pyx).T / py

    H_X = -np.dot(px, np.log(px))
    H_Y = -np.dot(py, np.log(py))
    IXY = 0
    for i, py_i in enumerate(py):
        IXY += py_i * entropy(px_y[:, i], px)

    log_betas = np.linspace(6, 1, 1000)
    betas = 2 ** log_betas

    ib_model = genIB.GenIB(dx, dy, pyx)
    dual_model = genIB.GenIB(dx, dy, pyx, dual=True)
    exp_model = genIB.GenIB(dx, dy, pyx, exp_dist=True, lambdas=lambdas, A=A)

    iters = np.zeros((3, betas.size))
    pt = np.zeros((3, betas.size, dt))
    py_t = np.zeros((3, betas.size, dx))
    pt_x = np.zeros((3, betas.size, dt, dx))
    IXT = np.zeros((3, betas.size))
    IYT = np.zeros((3, betas.size))

    # run BA for regIB, dualIB and expIB [respond to index 0,1,2 respectively]
    for i, b in enumerate(betas):
        iters[0, i], IXT[0, i], IYT[0, i] = ib_model.run_ba(b)
        iters[1, i], IXT[1, i], IYT[1, i] = dual_model.run_ba(b)
        iters[2, i], IXT[2, i], IYT[2, i] = exp_model.run_ba(b)
        pt[0, i, :] = ib_model.pt[ib_model.idx, :]
        pt[1, i, :] = dual_model.pt[dual_model.idx, :]
        pt[2, i, :] = exp_model.pt[exp_model.idx, :]
        py_t[0, i, :] = ib_model.py_t[ib_model.idx, 0, :]
        py_t[1, i, :] = dual_model.py_t[dual_model.idx, 0, :]
        py_t[2, i, :] = exp_model.py_t[exp_model.idx, 0, :]
        pt_x[0, i, :, :] = ib_model.pt_x[ib_model.idx, :, :]
        pt_x[1, i, :, :] = dual_model.pt_x[dual_model.idx, :, :]
        pt_x[2, i, :, :] = exp_model.pt_x[exp_model.idx, :, :]

    IXT = IXT / H_X
    IYT = IYT / IXY

    return IXT, IYT


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--type', type=int, required=True)
    # parser.add_argument('--n', type=float, required=True)
    # parser.add_argument('--out-dir', type=str, required=True)
    # args = parser.parse_args()

    # single_run(args)
    IXT, IYT = my_run()
