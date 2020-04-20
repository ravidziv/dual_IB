"""
 KL-based upper bound on entropy of mixture of Gaussians with covariance matrix var * I
    #  see Kolchinsky and Tracey, Estimating Mixture Entropy with Pairwise Distances, Entropy, 2017. Section 4.
    #  and Kolchinsky and Tracey, Nonlinear Information Bottleneck, 2017. Eq. 10
"""
import math as m
import tensorflow as tf
pi = tf.constant(m.pi)


@tf.function
def GMM_entropy(dist, var, d, bound='upper', weights=None, n=1):
    # computes bounds for the entropy of a homoscedastic Gaussian mixture model [Kolchinsky, 2017]
    # dist: a matrix of pairwise distances
    # log_var: the log-variance of the mixture components
    # d: number of dimensions
    # n: number of mixture components
    if bound is 'upper':
        dist_norm =  dist / (2.0 * var)  # uses the KL distance
    elif bound is 'lower':
        dist_norm =  dist / (8.0 * var)  # uses the Bhattacharyya distance
    normconst = (tf.cast(d, tf.float32)/2.0)*tf.math.log(2.*pi*var)
    lprobs = tf.reduce_logsumexp(-dist_norm, axis=1) -tf.math.log(tf.cast(n, tf.float32)) - normconst

    h = -tf.reduce_mean(lprobs)
    return tf.cast(d/2, tf.float64) + tf.cast(h, tf.float64)

@tf.function
def gaussian_entropy_np(d, var):
    # Entropy of a Gaussian distribution with 'd' dimensions and log variance 'log_var'
    log_d = tf.math.log(var)
    h = 0.5 * tf.cast(d, tf.float32) * (tf.math.log(2.0 * pi) + log_d+1)
    return h



@tf.function
def get_ixt_gmm(dist_matrix, input_dim = 0, n = 1, noisevar=None):
    """Get I(X;T) assumeing that p(t|x) is a gsussian with noise variance (nonlinear IB)."""
    H_T_given_X = gaussian_entropy_np(input_dim, noisevar)
    #dist_matrix = pairwise_distances(tf.cast(inputs, tf.float32))
    #H_T_lb = GMM_entropy(dist_matrix, noisevar, input_dim, 'lower')
    #Ixt_lb = H_T_lb - H_T_given_X
    H_T = GMM_entropy(dist_matrix, noisevar, input_dim, 'upper', n=n)
    ixt = tf.cast(H_T, tf.float64) - tf.cast(H_T_given_X, tf.float64)  # nonlinear IB upper bound on I(X;T)

    return tf.cast(ixt, tf.float64), tf.cast(H_T, tf.float64)