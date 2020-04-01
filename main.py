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

FLAGS = flags.FLAGS

flags.DEFINE_string('summary_path','logs/', '')
flags.DEFINE_string('checkpoint_path','log/', '')

flags.DEFINE_integer('x_dim',20, '')
flags.DEFINE_integer('num_train', int(1e5), '')
flags.DEFINE_integer('num_test', int(1e5), '')
flags.DEFINE_integer('y_dim', 2, '')

flags.DEFINE_float('alpha', 0.2, '')
flags.DEFINE_float('lambd', 10.6, '')

flags.DEFINE_float('lr', 1e-4, '')
flags.DEFINE_float('beta', 0.99, '')
flags.DEFINE_integer('num_epochs', 10000, '')
flags.DEFINE_integer('batch_size', 256, 'batch size for the main network')

flags.DEFINE_multi_integer('layer_width_generator', [10, 5], '')
flags.DEFINE_multi_integer('layers_width', [15, 10, 10], 'The width of the layers in main network')
flags.DEFINE_string('nonlin','relu', '')
flags.DEFINE_string('nonlin_dataset','relu', '')
flags.DEFINE_integer('num_of_save_steps', 200,'')
flags.DEFINE_integer('steps_per_epoch', 20000,'')

mlflow.tensorflow.autolog(1)

def train(train_ds, test_ds, lr, beta, num_epochs, input_shape, y_dim, nonlin, layers_width, steps_per_epoch=1):
    """Train the model and measure the information."""
    model = bulid_model(input_shape=input_shape, y_dim=y_dim, nonlin=nonlin, layers_width=layers_width)
    optimizer = tf.keras.optimizers.SGD(lr, beta)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=['accuracy'])

    checkpoint_path = mlflow.get_artifact_uri()+"/model/checkpoints/{}_cp"
    logdir = mlflow.get_artifact_uri()+ "/model"
    # Create a callback that saves the model's weights
    steps = np.unique(np.logspace(0, np.log10(num_epochs *steps_per_epoch), FLAGS.num_of_save_steps, dtype =np.int  ))
    #Save Cehckpoints of the model and csv file
    history = LoggerHistory(steps=steps, val_data=test_ds,train_data=train_ds,
                            file_name=mlflow.get_artifact_uri()+'/c_looger.csv', checkpoint_path=checkpoint_path)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    print (steps)
    r = model.fit(train_ds, steps_per_epoch=steps_per_epoch,
              verbose    = 1,
              epochs     = num_epochs,
            callbacks = [tensorboard_callback, history ])


def main(argv):
    with mlflow.start_run():
        strategy = tf.distribute.MirroredStrategy()
        BATCH_SIZE = FLAGS.batch_size * strategy.num_replicas_in_sync
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        train_ds, test_ds, py, py_x, xs, px, A, lambd= create_dataset(FLAGS.num_train, FLAGS.num_test,
                                           FLAGS.x_dim, FLAGS.layer_width_generator,  FLAGS.nonlin_dataset, batch_size=BATCH_SIZE,
                                           lambd_factor=FLAGS.lambd,
                                           alpha=FLAGS.alpha)
        #Save params
        for key in FLAGS.flag_values_dict():
            try:
                value = FLAGS.get_flag_value(key, None)
                mlflow.log_param(key, value)
            except:
                continue

        with strategy.scope():
            train(train_ds=train_ds, test_ds=test_ds, lr=FLAGS.lr, beta=FLAGS.beta, num_epochs=FLAGS.num_epochs,
              input_shape=FLAGS.x_dim, y_dim=FLAGS.y_dim , nonlin=FLAGS.nonlin, layers_width= FLAGS.layers_width,
                  steps_per_epoch = FLAGS.steps_per_epoch)

if __name__ == "__main__":
    app.run(main)