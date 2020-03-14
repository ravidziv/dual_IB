
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

FLAGS = flags.FLAGS

flags.DEFINE_string('summary_path','logs/', '')
flags.DEFINE_string('checkpoint_path','log/', '')

from absl import flags
from absl import app
flags.DEFINE_integer('x_dim',10, '')
flags.DEFINE_integer('num_train', 100, '')
flags.DEFINE_integer('num_test', 100, '')
flags.DEFINE_integer('input_shape', 10, '')
flags.DEFINE_integer('y_dim', 2, '')

flags.DEFINE_float('alpha', 0.15, '')
flags.DEFINE_float('lambd', 10.6, '')

flags.DEFINE_float('lr', 1e-2, '')
flags.DEFINE_float('beta', 0.99, '')
flags.DEFINE_integer('num_epochs', 1000, '')
flags.DEFINE_integer('batch_per_epoch', 100, '')
flags.DEFINE_integer('num_iterations_to_print', 5, '')
flags.DEFINE_integer('num_of_epochs_inf_labels', 5, '')
flags.DEFINE_integer('num_of_samples', 10, '')
flags.DEFINE_integer('num_of_bins', 30, 'The number of bins for the bins estimator')
flags.DEFINE_integer('batch_size', 128, 'batch size for the main network')

flags.DEFINE_float('noisevar', 1e-1, '')
flags.DEFINE_float('lr_labels', 5e-2, '')
flags.DEFINE_multi_integer('layer_widths', [10, 5], '')
flags.DEFINE_multi_integer('num_of_clusters', [3, 3, 3, 3], 'The width of the layers in the random network')
flags.DEFINE_multi_integer('layers_width', [10, 10, 10], 'The width of the layers in main network')
flags.DEFINE_string('nonlin','tanh', '')
flags.DEFINE_string('nonlin_dataset','Relu', '')

def bulid_model(input_shape, y_dim=2, nonlin = 'tanh', layers_width=[]):
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Dense(layers_width[0], input_shape=(input_shape,), activation=nonlin))
  for i in range(1, len(layers_width)):
    model.add(tf.keras.layers.Dense(layers_width[i], activation=nonlin))
  model.add(tf.keras.layers.Dense(y_dim))
  return model

mlflow.tensorflow.autolog(1)

def train(train_ds, test_ds, lr, beta, num_epochs, batch_per_epoch,
          num_iterations_to_print, noisevar, py, py_x, xs, px, A, lambd,
          lr_labels, num_of_clusters, num_of_epochs_inf_labels, beta_func, num_of_samples,
          num_of_bins, input_shape, y_dim, nonlin, layers_width):
    """Train the model and measure the information."""
    model = bulid_model(input_shape=input_shape, y_dim=y_dim, nonlin=nonlin, layers_width=layers_width)
    optimizer = tf.keras.optimizers.SGD(lr, beta)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    summary_writer = tf.summary.create_file_writer(FLAGS.summary_path)
    #ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    #manager = tf.train.CheckpointManager(ckpt, FLAGS.checkpoint_path, max_to_keep=3)
    #ckpt.restore(manager.latest_checkpoint)
    print ('1111111',mlflow.get_artifact_uri())
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=['accuracy'])
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        mlflow.get_artifact_uri() +'/model', verbose=2, save_best_only=False,
        save_weights_only=False, period=1
    )
    checkpoint_path = mlflow.get_artifact_uri()+"/model/checkpoints/{epoch}_cp.ckpt"
    #checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_best_only=True,
                                                     save_weights_only=False,
                                                     verbose=1)


    r = model.fit(train_ds, validation_data=test_ds,
              verbose    = 2,
              epochs     =num_epochs,
                  callbacks = [cp_callback])

def main(argv):
    with mlflow.start_run():
        train_ds, test_ds, py, py_x, xs, px, A, lambd= create_dataset(FLAGS.num_train, FLAGS.num_test,
                                           FLAGS.x_dim, FLAGS.layer_widths,  FLAGS.nonlin_dataset, batch_size=FLAGS.batch_size,
                                           lambd_factor=FLAGS.lambd,
                                           alpha=FLAGS.alpha)
        train(train_ds, test_ds, FLAGS.lr, FLAGS.beta, FLAGS.num_epochs,
              FLAGS.batch_per_epoch,
              FLAGS.num_iterations_to_print, FLAGS.noisevar, py=py, py_x=py_x, xs=xs, px=px, A=A, lambd=lambd,
              lr_labels=FLAGS.lr_labels, num_of_clusters=FLAGS.num_of_clusters,
              num_of_epochs_inf_labels=FLAGS.num_of_epochs_inf_labels, beta_func=[],
              num_of_samples=FLAGS.num_of_samples, num_of_bins=FLAGS.num_of_bins,
              input_shape=FLAGS.input_shape, y_dim=FLAGS.y_dim , nonlin=FLAGS.nonlin, layers_width= FLAGS.layers_width
              )

if __name__ == "__main__":
    app.run(main)