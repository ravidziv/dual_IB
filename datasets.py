import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def bulid_model(layer_widths, y_dim=2, nonlin='relu'):
  model = tf.keras.Sequential()
  for i, width in enumerate(layer_widths):
      model.add(tf.keras.layers.Dense(width, activation='relu'))
  model.add(tf.keras.layers.Dense(y_dim, activation='linear'))
  return model


def get_data(probs, num_test, py_x_samp, x_samp, xt_samp, py_xt_sampe, batch_size):
    with tf.device('/cpu:0'):
        py = tfp.distributions.Bernoulli(probs=tf.cast(probs[0], tf.float64))
        px = tfp.distributions.Categorical(probs=np.ones((num_test))/num_test)
        train_ds = tf.data.Dataset.from_tensor_slices((x_samp, tf.transpose(py_x_samp))).shuffle(buffer_size=100).with_options(tf.data.Options())
        test_ds = tf.data.Dataset.from_tensor_slices((xt_samp, tf.transpose(py_xt_sampe))).shuffle(buffer_size=100).with_options(tf.data.Options())
        train_ds = train_ds.batch(batch_size).repeat()
        test_ds = test_ds.batch(batch_size).repeat()
    return train_ds, test_ds, py, px

def create_dataset_np(num_train, num_test, x_dim, layer_widths, nonlin, lambd_factor = 2.6,
                   alpha=0.025, batch_size=128, r=5):
    # Sample random gaussian xs
    tf.random.set_seed(1)
    with tf.device('/cpu:0'):
        x = tf.random.normal(shape = (num_train + num_test, x_dim))
        # Pass it through the network
        model = bulid_model(layer_widths, nonlin=nonlin, y_dim=r)
        output = model(x)
        # Change lambd for determenistic level of p(y|x)
        A = tf.concat([tf.cast(output, tf.float64), alpha*tf.ones((len(output),1), dtype=tf.float64)], axis=1)
        lambd = lambd_factor/tf.stack([tf.ones(r+1), -tf.ones(r+1)])
        py_x = tf.exp(tf.einsum('ji,ki->kj', tf.cast(A, tf.float64), tf.cast(lambd, tf.float64)))
        # Nromalized it
        Zy_x = tf.reduce_sum(py_x, axis=0)
        py_x_normalize = (py_x / Zy_x[None,:])
        # Divided to train/test
        x_samp, xt_samp = x[:num_train, :], x[num_train:, :]
        py_x_samp, py_xt_sampe = py_x_normalize[:,:num_train], py_x_normalize[:,num_train:]
        probs = tf.reduce_sum(py_x_samp, axis=1)
        probs = probs / np.sum(probs)
    A = tf.cast(tf.transpose(A[num_train:]), tf.float64)
    lambd = tf.cast(tf.transpose(lambd), tf.float64)
    return x_samp.numpy(), py_x_samp.numpy(), xt_samp.numpy(), py_xt_sampe.numpy(), probs.numpy() , xt_samp.numpy(),\
           A.numpy(), lambd.numpy()

def create_dataset(num_train, num_test, x_dim, layer_widths, nonlin, lambd_factor = 2.6,
                   alpha=0.025, batch_size=128, r=5, train_batch_size=128):
    """Create a exponential dataset by ranodm network"""
    # Sample random gaussian xs
    tf.random.set_seed(1)
    with tf.device('/cpu:0'):
        x = tf.random.normal(shape = (num_train + num_test, x_dim))
        # Pass it through the network
        model = bulid_model(layer_widths, nonlin=nonlin, y_dim=r)
        output = model(x)
        # Change lambd for determenistic level of p(y|x)
        A = tf.concat([tf.cast(output, tf.float64), alpha*tf.ones((len(output),1), dtype=tf.float64)], axis=1)
        lambd = lambd_factor/tf.stack([tf.ones(r+1), -tf.ones(r+1)])
        py_x = tf.exp(tf.einsum('ji,ki->kj', tf.cast(A, tf.float64), tf.cast(lambd, tf.float64)))
        # Nromalized it
        #Zy_x = tf.reduce_sum(py_x, axis=0)
        indices = tf.where(tf.math.is_inf(py_x))
        py_x_ing = tf.tensor_scatter_nd_update(tf.cast(py_x, tf.float64), indices,
                                               tf.cast(tf.ones((indices.shape[0])), tf.float64))
        Zy_x = tf.reduce_sum(py_x_ing, axis=0)
        py_x_normalize = (py_x_ing / Zy_x[None,:])
        #py_x_normalize = tf.cast(py_x_normalize, tf.float16)

        # Divided to train/test
        x_samp, xt_samp = x[:num_train, :], x[num_train:, :]
        py_x_samp, py_xt_sampe = py_x_normalize[:,:num_train], py_x_normalize[:,num_train:]
        probs = tf.reduce_sum(py_x_normalize, axis=1)
        probs = probs / np.sum(probs)
        py = tfp.distributions.Categorical(probs=tf.cast(tf.transpose(probs), tf.float32))
        px = tfp.distributions.Categorical(probs=tf.cast(np.ones((num_train + num_test))/(num_train + num_test), tf.float32))
        py_x = tfp.distributions.Categorical(probs=tf.cast(tf.transpose(py_x_normalize), tf.float32))
        pyx_s = py_x.probs * px.probs[:,None]
        px_y_s =tf.transpose(pyx_s) / py.probs[:, None]
        px_y = tfp.distributions.Categorical(probs=tf.cast(px_y_s, tf.float32))
        ixy = tf.reduce_sum(py.probs * tfp.distributions.kl_divergence(px_y, px))
    A = tf.cast(tf.transpose(A), tf.float32)
    lambd = tf.cast(tf.transpose(lambd), tf.float32)

    train_ds = tf.data.Dataset.from_tensor_slices((x_samp, tf.transpose(py_x_samp))).shuffle(buffer_size=1000)
    test_ds = tf.data.Dataset.from_tensor_slices((xt_samp, tf.transpose(py_xt_sampe))).shuffle(buffer_size=1000)
    train_ds = train_ds.batch(train_batch_size).repeat()
    test_ds = test_ds.batch(batch_size).repeat()
    return train_ds, test_ds, py,py_x, x, px, A, lambd, ixy