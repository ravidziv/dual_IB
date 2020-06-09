"""data sets (FasionMNIST, CIFAR10/100) including confusion matrices"""
from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def fasion_mnist_preprocessing(image):
    """Normalizes images: `uint8` -> `float32`."""
    # image = tf.repeat(image, 0)
    image = tf.repeat(image, 3, 3)
    return tf.cast(image, tf.float32) / 255.


def cifar_preprocessing(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean = [125.3, 123.0, 113.9]
    std = [63.0, 62.1, 66.7]
    for i in range(3):
        x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
    return x_train, x_test


def cifar100_preprocessing(x_train, x_test):
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    for i in range(3):
        x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]

    return x_train, x_test


def load_cifar_data(num_class=10, batch_size=128, IMG_ROWS=32, IMG_COLS=32, IMG_CHANNELS=3, num_of_train=-1):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = cifar_preprocessing(x_train, x_test)
    x_train = x_train[:num_of_train]
    y_train = y_train[:num_of_train]
    ds_test = tf.data.Dataset.from_tensor_slices((x_test, np.reshape(y_test, (-1,))))
    ds_test = ds_test.batch(batch_size, drop_remainder=True).repeat()
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
    confusion_matrix = [[828, 13, 12, 11, 18, 0, 2, 4, 85, 27],
                        [10, 910, 0, 5, 1, 1, 0, 1, 11, 61],
                        [47, 1, 708, 64, 88, 14, 63, 4, 8, 3],
                        [3, 4, 16, 768, 33, 93, 50, 19, 4, 10],
                        [10, 0, 39, 43, 788, 12, 57, 43, 6, 2],
                        [2, 0, 10, 137, 29, 777, 8, 33, 0, 4],
                        [7, 2, 10, 54, 29, 7, 888, 1, 1, 1],
                        [24, 2, 14, 39, 76, 17, 4, 818, 2, 4],
                        [27, 13, 0, 7, 3, 0, 3, 0, 933, 14],
                        [19, 64, 1, 7, 2, 1, 1, 0, 18, 887]]
    confusion_matrix = np.array(confusion_matrix)
    confusion_matrix = confusion_matrix.astype(np.float64) / np.sum(confusion_matrix, axis=1)
    return ds_train, ds_test, step_per_epoch, steps_per_epoch_validation, confusion_matrix


def load_mnist_data(batch_size=128, num_epochs=25, num_of_train=-1, step_per_epoch=50):
    train, test = tf.keras.datasets.fashion_mnist.load_data()
    x_train, y_train = train
    x_test, y_test = test
    x_train = x_train[:num_of_train]
    y_train = y_train[:num_of_train]
    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)
    x_test = fasion_mnist_preprocessing(x_test)

    x_train = fasion_mnist_preprocessing(x_train)
    datagen = ImageDataGenerator(horizontal_flip=True,
                                 width_shift_range=0.125, height_shift_range=0.125, fill_mode='reflect')

    datagen.fit(x_train, augment=True)
    part_f = partial(datagen.flow, batch_size=batch_size, shuffle=True)
    ds_train = tf.data.Dataset.from_generator(
        part_f, args=[x_train, np.reshape(y_train, (-1,))],
        output_types=(tf.float32, tf.int32),
        output_shapes=((None, 28, 28, 1), [None]))
    ds_test = tf.data.Dataset.from_tensor_slices((x_test, np.reshape(y_test, (-1,))))
    ds_test = ds_test.batch(batch_size * 10, drop_remainder=True).repeat()
    steps_per_epoch_validation = len(x_test) // batch_size + 1
    confusion_matrix = [[783, 0, 15, 12, 2, 1, 73, 0, 6, 0],
                        [0, 876, 2, 8, 1, 0, 3, 0, 0, 0],
                        [12, 1, 811, 8, 35, 0, 38, 0, 0, 0],
                        [20, 4, 10, 836, 21, 0, 24, 0, 1, 0],
                        [0, 0, 65, 20, 776, 0, 52, 0, 0, 0],
                        [0, 0, 0, 0, 0, 884, 0, 10, 0, 6],
                        [88, 1, 44, 19, 50, 0, 683, 0, 4, 0],
                        [0, 0, 0, 0, 0, 5, 0, 888, 0, 17],
                        [4, 1, 1, 1, 4, 2, 3, 1, 868, 1],
                        [0, 0, 0, 0, 0, 4, 0, 18, 1, 876]]
    confusion_matrix = np.array(confusion_matrix)
    confusion_matrix = confusion_matrix.astype(np.float64) / np.sum(confusion_matrix, axis=1)
    return ds_train, ds_test, step_per_epoch, steps_per_epoch_validation, confusion_matrix


def load_fasion_mnist_data(batch_size=128, num_epochs=25, num_of_train=-1, step_per_epoch=50):
    train, test = tf.keras.datasets.fashion_mnist.load_data()
    x_train, y_train = train
    x_test, y_test = test
    x_train = x_train[:num_of_train]
    y_train = y_train[:num_of_train]
    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)
    x_test = fasion_mnist_preprocessing(x_test)

    x_train = fasion_mnist_preprocessing(x_train)
    datagen = ImageDataGenerator(horizontal_flip=True,
                                 width_shift_range=0.125, height_shift_range=0.125, fill_mode='reflect')

    datagen.fit(x_train, augment=True)
    part_f = partial(datagen.flow, batch_size=batch_size, shuffle=True)
    ds_train = tf.data.Dataset.from_generator(
        part_f, args=[x_train, np.reshape(y_train, (-1,))],
        output_types=(tf.float32, tf.int32),
        output_shapes=((None, 28, 28, 3), [None]))
    # ds_train = tf.data.Dataset.from_tensor_slices((x_train, np.reshape(y_train, (-1,))))
    ds_test = tf.data.Dataset.from_tensor_slices((x_test, np.reshape(y_test, (-1,))))
    ds_test = ds_test.batch(batch_size * 10, drop_remainder=True).repeat()
    steps_per_epoch_validation = len(x_test) // batch_size + 1
    confusion_matrix = [[783, 0, 15, 12, 2, 1, 73, 0, 6, 0],
                        [0, 876, 2, 8, 1, 0, 3, 0, 0, 0],
                        [12, 1, 811, 8, 35, 0, 38, 0, 0, 0],
                        [20, 4, 10, 836, 21, 0, 24, 0, 1, 0],
                        [0, 0, 65, 20, 776, 0, 52, 0, 0, 0],
                        [0, 0, 0, 0, 0, 884, 0, 10, 0, 6],
                        [88, 1, 44, 19, 50, 0, 683, 0, 4, 0],
                        [0, 0, 0, 0, 0, 5, 0, 888, 0, 17],
                        [4, 1, 1, 1, 4, 2, 3, 1, 868, 1],
                        [0, 0, 0, 0, 0, 4, 0, 18, 1, 876]]
    confusion_matrix = np.array(confusion_matrix)
    confusion_matrix = confusion_matrix.astype(np.float64) / np.sum(confusion_matrix, axis=1)
    return ds_train, ds_test, step_per_epoch, steps_per_epoch_validation, confusion_matrix


def load_cifar100_data(batch_size=128, num_epochs=200, num_of_train=-1):
    train, test = tf.keras.datasets.cifar100.load_data()
    x_train, y_train = train
    x_test, y_test = test
    x_train, x_test = cifar100_preprocessing(x_train, x_test)

    x_train = x_train[:num_of_train]
    y_train = y_train[:num_of_train]
    datagen = ImageDataGenerator(horizontal_flip=True,
                                 width_shift_range=0.125, height_shift_range=0.125, fill_mode='reflect')

    datagen.fit(x_train, augment=True)
    part_f = partial(datagen.flow, batch_size=batch_size, shuffle=True)

    ds_train = tf.data.Dataset.from_generator(
        part_f, args=[x_train, np.reshape(y_train, (-1,))],
        output_types=(tf.float32, tf.int32),
        output_shapes=((None, 32, 32, 3), [None]))

    ds_test = tf.data.Dataset.from_tensor_slices((x_test, np.reshape(y_test, (-1,))))
    ds_test = ds_test.batch(batch_size * 5, drop_remainder=True).repeat()
    step_per_epoch = len(x_train) // batch_size + 1
    steps_per_epoch_validation = len(x_test) // batch_size + 1
    confusion_matrix = tf.random.normal(shape=(100, 100), mean=np.eye(100), stddev=0.0001)
    confusion_matrix = np.abs(confusion_matrix) / np.sum(np.abs(confusion_matrix), axis=1)
    return ds_train, ds_test, step_per_epoch, steps_per_epoch_validation, confusion_matrix
