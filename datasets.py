import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def bulid_model(layer_widths, y_dim=2, nonlin='relu'):
  model = tf.keras.Sequential()
  for i, width in enumerate(layer_widths):
      model.add(tf.keras.layers.Dense(width, activation='relu'))
  model.add(tf.keras.layers.Dense(y_dim, activation='linear'))
  return model


def create_dataset(num_train, num_test, x_dim, layer_widths, nonlin, lambd_factor = 2.6,
                   alpha=0.025, batch_size=128, r=5):
    # Sample random gaussian xs
    tf.random.set_seed(1)
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
    probs = np.sum(py_x_samp.numpy(), axis=1)
    probs = probs / np.sum(probs)
    py = tfp.distributions.Categorical(probs=np.array(probs))
    px = tfp.distributions.Categorical(probs=np.ones((num_test))/num_test)
    train_ds = tf.data.Dataset.from_tensor_slices((x_samp, tf.transpose(py_x_samp))).shuffle(buffer_size=1000)
    test_ds = tf.data.Dataset.from_tensor_slices((xt_samp, tf.transpose(py_xt_sampe))).shuffle(buffer_size=1000)

    train_ds = train_ds.batch(batch_size).repeat()
    test_ds = test_ds.batch(batch_size).repeat()
    return train_ds, test_ds, py, tf.transpose(py_xt_sampe), xt_samp, px, A[num_train:], lambd