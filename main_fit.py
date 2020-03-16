"""Train and save a network during the training process"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging, os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from absl import flags
from absl import app
from datasets import create_dataset
from network_utils import bulid_model
FLAGS = flags.FLAGS

flags.DEFINE_string('summary_path','logs/', '')
flags.DEFINE_string('checkpoint_path','log/', '')

from absl import flags
flags.DEFINE_integer('x_dim',10, '')
flags.DEFINE_integer('num_train', 10000, '')
flags.DEFINE_integer('num_test', 2000, '')
flags.DEFINE_integer('input_shape', 10, '')
flags.DEFINE_integer('y_dim', 2, '')

flags.DEFINE_float('alpha', 1.15, '')
flags.DEFINE_float('lambd', 10.6, '')

flags.DEFINE_float('lr', 1e-2, '')
flags.DEFINE_float('beta', 0.99, '')
flags.DEFINE_integer('num_epochs', 1000, '')
flags.DEFINE_integer('batch_size', 128, 'batch size for the main network')

flags.DEFINE_multi_integer('layer_widths', [10, 5], '')
flags.DEFINE_multi_integer('layers_width', [10, 10, 10], 'The width of the layers in main network')
flags.DEFINE_string('nonlin','tanh', '')
flags.DEFINE_string('nonlin_dataset','Relu', '')


mlflow.tensorflow.autolog(1)

def train(train_ds, test_ds, lr, beta, num_epochs, input_shape, y_dim, nonlin, layers_width):
    """Train the model and measure the information."""
    model = bulid_model(input_shape=input_shape, y_dim=y_dim, nonlin=nonlin, layers_width=layers_width)
    optimizer = tf.keras.optimizers.SGD(lr, beta)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=['accuracy'])
    checkpoint_path = mlflow.get_artifact_uri()+"/model/checkpoints/{epoch}_cp"
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_best_only=False,
                                                     save_weights_only=False,
                                                     verbose=0)
    r = model.fit(train_ds, validation_data=test_ds,
              verbose    = 2,
              epochs     = num_epochs,
            callbacks = [cp_callback])

def main(argv):
    with mlflow.start_run():
        train_ds, test_ds, py, py_x, xs, px, A, lambd= create_dataset(FLAGS.num_train, FLAGS.num_test,
                                           FLAGS.x_dim, FLAGS.layer_widths,  FLAGS.nonlin_dataset, batch_size=FLAGS.batch_size,
                                           lambd_factor=FLAGS.lambd,
                                           alpha=FLAGS.alpha)
        #Save params
        for key in FLAGS.flag_values_dict():
            try:
                value = FLAGS.get_flag_value(key, None)
                mlflow.log_param(key, value)
            except:
                continue
        train(train_ds=train_ds, test_ds=test_ds, lr=FLAGS.lr, beta=FLAGS.beta, num_epochs=FLAGS.num_epochs,
              input_shape=FLAGS.input_shape, y_dim=FLAGS.y_dim , nonlin=FLAGS.nonlin, layers_width= FLAGS.layers_width
              )

if __name__ == "__main__":
    app.run(main)