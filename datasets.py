import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import load_cifar10 as cifar10
from functools import partial
import matplotlib.pyplot as plt

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
    A = tf.cast(tf.transpose(A), tf.float64)
    lambd = tf.cast(tf.transpose(lambd), tf.float64)

    train_ds = tf.data.Dataset.from_tensor_slices((x_samp, tf.transpose(py_x_samp))).shuffle(buffer_size=1000)
    test_ds = tf.data.Dataset.from_tensor_slices((xt_samp, tf.transpose(py_xt_sampe))).shuffle(buffer_size=1000)
    train_ds = train_ds.batch(train_batch_size).repeat()
    test_ds = test_ds.batch(batch_size).repeat()
    return train_ds, test_ds, py,py_x, x, px, A, lambd, ixy




def mnist_preprocessing(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  # image = tf.repeat(image, 0)
  image = tf.repeat(image, 3, 2)
  return tf.cast(image, tf.float32) / 255., label


def cifar_preprocessing(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean = [125.3, 123.0, 113.9]
    std  = [63.0,  62.1,  66.7]
    for i in range(3):
        x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
        x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]

    return x_train, x_test


def load_cifar_data(num_class=10, batch_size=128, IMG_ROWS=32, IMG_COLS=32, IMG_CHANNELS=3, num_of_train=-1):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = cifar_preprocessing(x_train, x_test)
    x_train = x_train[:num_of_train]
    y_train = y_train[:num_of_train]
    # ds_train = tf.data.Dataset.from_tensor_slices((x_train, np.reshape(y_train, (-1,))))
    # ds_train = ds_train.batch(batch_size, drop_remainder=True)
    ds_test = tf.data.Dataset.from_tensor_slices((x_test, np.reshape(y_test, (-1,))))
    ds_test = ds_test.batch(batch_size, drop_remainder=True)
    datagen = ImageDataGenerator(horizontal_flip=True,
                                 width_shift_range=0.125, height_shift_range=0.125, fill_mode='reflect')

    datagen.fit(x_train, augment=True)
    part_f = partial(datagen.flow, batch_size=batch_size, shuffle=True)

    ds_train = tf.data.Dataset.from_generator(
        part_f, args=[x_train, np.reshape(y_train, (-1,))],
        output_types=(tf.float32, tf.int32),
        output_shapes=((None, IMG_ROWS, IMG_COLS, IMG_CHANNELS), [None]))
    step_per_epoch = len(x_train) // batch_size + 1
    steps_per_epoch_validation = len(x_test) // batch_size + 1
    confusion_matrix = np.array([[0.82232, 0.00238, 0.021, 0.00069, 0.00108, 0, 0.00017, 0.00019, 0.1473, 0.00489],
                                 [0.00233, 0.83419, 0.00009, 0.00011, 0, 0.00001, 0.00002, 0, 0.00946, 0.15379],
                                 [0.03139, 0.00026, 0.76082, 0.0095, 0.07764, 0.01389, 0.1031, 0.00309, 0.00031, 0],
                                 [0.00096, 0.0001, 0.00273, 0.69325, 0.00557, 0.28067, 0.01471, 0.00191, 0.00002,
                                  0.0001],
                                 [0.00199, 0, 0.03866, 0.00542, 0.83435, 0.01273, 0.02567, 0.08066, 0.00052, 0.00001],
                                 [0, 0.00004, 0.00391, 0.2498, 0.00531, 0.73191, 0.00477, 0.00423, 0.00001, 0],
                                 [0.00067, 0.00008, 0.06303, 0.05025, 0.0337, 0.00842, 0.8433, 0, 0.00054, 0],
                                 [0.00157, 0.00006, 0.00649, 0.00295, 0.13058, 0.02287, 0, 0.83328, 0.00023, 0.00196],
                                 [0.1288, 0.01668, 0.00029, 0.00002, 0.00164, 0.00006, 0.00027, 0.00017, 0.83385,
                                  0.01822],
                                 [0.01007, 0.15107, 0, 0.00015, 0.00001, 0.00001, 0, 0.00048, 0.02549, 0.81273]])
    return ds_train, ds_test, step_per_epoch, steps_per_epoch_validation, confusion_matrix


def load_mnist_data(batch_size=128, num_epochs = 25):
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    # Create Data
    ds_train = ds_train.map(
        mnist_preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(batch_size, drop_remainder=True)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE).repeat(num_epochs)
    ds_test = ds_test.map(
        mnist_preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(batch_size, drop_remainder=True)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
    step_per_epoch = ds_info.splits['train'].num_examples//batch_size +1
    return ds_train, ds_test, step_per_epoch