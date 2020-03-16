import numpy as np


def calc_entropy_for_specipic_t(current_ts, px_i):
    """Calc entropy for specipic t"""
    b2 = np.ascontiguousarray(current_ts).view(
        np.dtype((np.void, current_ts.dtype.itemsize * current_ts.shape[1])))
    unique_array, unique_inverse_t, unique_counts = \
        np.unique(b2, return_index=False, return_inverse=True, return_counts=True)
    p_current_ts = unique_counts / float(sum(unique_counts))
    p_current_ts = np.asarray(p_current_ts, dtype=np.float64).T
    H2X = px_i * (-np.sum(p_current_ts * np.log2(p_current_ts)))
    return H2X


def calc_condtion_entropy(px, t_data, unique_inverse_x):
    # Condition entropy of t given x
    H2X_array = [calc_entropy_for_specipic_t(t_data[unique_inverse_x == i, :], px[i])
                                   for i in range(px.shape[0])]
    H2X = np.sum(np.array(H2X_array))
    return H2X


ent = lambda pt: -np.sum(pt * np.log2(pt), axis=0)

def calc_information_from_mat(px, py, pt, data, unique_inverse_x, py_t):
    """Calculate the MI based on binning of the data"""
    Ht = ent(pt)
    hy = ent(py)
    HYT =np.sum(pt*(ent(py_t)))
    #H2X = calc_condtion_entropy(px, data, unique_inverse_x)
    IY = hy - HYT
    return IY, Ht

def extract_probs(label, x):
	"""calculate the probabilities of the given data and labels p(x), p(y) and (y|x)"""
	pys = np.sum(label, axis=0) / float(label.shape[0])

	b = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
	unique_array, unique_indices, unique_inverse_x, unique_counts = \
		np.unique(b, return_index=True, return_inverse=True, return_counts=True)
	pxs = unique_counts / float(np.sum(unique_counts))
	b_y = np.ascontiguousarray(label).view(np.dtype((np.void, label.dtype.itemsize * label.shape[1])))
	unique_array_y, unique_indices_y, unique_inverse_y, unique_counts_y = \
		np.unique(b_y, return_index=True, return_inverse=True, return_counts=True)
	return pxs, pys, unique_inverse_x, unique_inverse_y
