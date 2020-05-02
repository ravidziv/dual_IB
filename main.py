"""Train and save a network during the training process"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import logging, os
from csv import writer
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from absl import flags
from absl import app
from datasets import create_dataset
from network_utils import bulid_model
from utils import LoggerHistory
from absl import flags
import scipy.io as sio
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'symmetric', 'Which dataset to load')
flags.DEFINE_string('summary_path','logs/', '')
flags.DEFINE_string('checkpoint_path','log/', '')

flags.DEFINE_integer('x_dim', 12, '')
flags.DEFINE_integer('num_train', int(1e5), '')
flags.DEFINE_integer('num_test', int(1e4), '')
flags.DEFINE_integer('y_dim', 2, '')

flags.DEFINE_float('alpha', 0.2, '')
flags.DEFINE_float('lambd', 10.6, '')

flags.DEFINE_float('lr', 0.0004, '')
flags.DEFINE_float('beta', 0.99, '')
flags.DEFINE_integer('num_epochs', 10000, '')
flags.DEFINE_integer('batch_size', 256, 'batch size for the main network')

flags.DEFINE_multi_integer('layer_width_generator', [10, 5], '')
flags.DEFINE_multi_integer('layers_width', [15,5,4,3], 'The width of the layers in main network')
flags.DEFINE_string('nonlin', 'relu', 'tanh or relu')
flags.DEFINE_string('nonlin_dataset', 'tanh', '')
flags.DEFINE_integer('num_of_save_steps', 100,'')
flags.DEFINE_integer('steps_per_epoch', 140,'')

mlflow.tensorflow.autolog(1)


def train(train_ds, test_ds, lr, beta, num_epochs, input_shape, y_dim, nonlin, layers_width, steps_per_epoch=1,
          loss_fn=[]):
    """Train the model and measure the information."""
    model = bulid_model(input_shape=input_shape, y_dim=y_dim, nonlin=nonlin, layers_width=layers_width)
    #optimizer = tf.keras.optimizers.SGD(lr, beta)
    optimizer = tf.keras.optimizers.Adam(lr)
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=['accuracy'])

    checkpoint_path = mlflow.get_artifact_uri()+"/model1/checkpoints/{}_cp"
    #logdir = mlflow.get_artifact_uri()+ "/model1"
    # Create a callback that saves the model's weights
    #steps = np.unique(np.logspace(0, np.log10(num_epochs *steps_per_epoch), FLAGS.num_of_save_steps, dtype =np.int  ))
    steps = np.concatenate((np.arange(0, 20), np.arange(20,100, 20), np.arange(100, 1000, 50), np.arange(1000, num_epochs, 100)))
    #Save Cehckpoints of the model and csv file
    history = LoggerHistory(steps=steps, val_data=test_ds,train_data=train_ds,
                            file_name=mlflow.get_artifact_uri()+'/c_looger.csv', checkpoint_path=checkpoint_path)
    #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    print (steps)
    r = model.fit(train_ds, steps_per_epoch=steps_per_epoch, validation_data=test_ds,
                  verbose=2,
                  epochs=num_epochs,
                  callbacks = [history ])
    plt.plot(r.history['val_loss'])
    plt.show()
    print (r)


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    image = tf.keras.backend.flatten(image)
    return tf.cast(image, tf.float32) / 255., label

def main(argv):

        strategy = tf.distribute.MirroredStrategy()
        BATCH_SIZE = FLAGS.batch_size * strategy.num_replicas_in_sync
        #BATCH_SIZE = FLAGS.batch_size * 1
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        if FLAGS.dataset == 'random_network':
            ds_train, ds_test, _, _, _, _, _, _ = create_dataset(FLAGS.num_train, FLAGS.num_test,
                                           FLAGS.x_dim, FLAGS.layer_width_generator,  FLAGS.nonlin_dataset, batch_size=BATCH_SIZE,
                                           lambd_factor=FLAGS.lambd,train_batch_size=BATCH_SIZE,
                                           alpha=FLAGS.alpha)
            loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            layers_width = FLAGS.layers_width
        elif FLAGS.dataset == 'mnist':
            (ds_train, ds_test), ds_info = tfds.load(
                'mnist',
                split=['train', 'test'],
                shuffle_files=True,
                as_supervised=True,
                with_info=True,
            )

            ds_train = ds_train.map(
                normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            ds_train = ds_train.cache()
            ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
            ds_train = ds_train.batch(128)
            ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE).repeat()
            ds_test = ds_test.map(
                normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            ds_test = ds_test.batch(128)
            ds_test = ds_test.cache()
            ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
            layers_width = [500, 128, 40]
        elif FLAGS.dataset == 'symmetric':
            dict = sio.loadmat('var_u.mat')
            Xs = dict['F']
            Ys = 1 - tf.one_hot(dict['y'][0], 2).numpy()
            X_train, X_test, y_train, y_test = train_test_split(
                Xs, Ys, test_size=0.2, random_state=42)
            ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
            ds_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))

            # ds_train = ds_train.map(
            #    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            ds_train = ds_train.cache()
            ds_train = ds_train.shuffle(X_train.shape[0])
            ds_train = ds_train.batch(128)
            ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE).repeat()
            ds_test = ds_test.batch(128)
            ds_test = ds_test.cache()
            ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
            loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

            layers_width = [10, 7, 5, 4]

        #Save params
        for key in FLAGS.flag_values_dict():
            try:
                value = FLAGS.get_flag_value(key, None)
                mlflow.log_param(key, value)
            except:
                continue

        # with strategy.scope():
        train(train_ds=ds_train, test_ds=ds_test, lr=FLAGS.lr, beta=FLAGS.beta, num_epochs=FLAGS.num_epochs,
              input_shape=FLAGS.x_dim, y_dim=FLAGS.y_dim, nonlin=FLAGS.nonlin, layers_width=layers_width,
              steps_per_epoch=FLAGS.steps_per_epoch, loss_fn=loss_fn)

if __name__ == "__main__":
    app.run(main)