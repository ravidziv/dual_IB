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
FLAGS = flags.FLAGS

flags.DEFINE_string('summary_path','logs/', '')
flags.DEFINE_string('checkpoint_path','log/', '')

from absl import flags
flags.DEFINE_integer('x_dim',20, '')
flags.DEFINE_integer('num_train', int(1e4), '')
flags.DEFINE_integer('num_test', int(1e6), '')
flags.DEFINE_integer('y_dim', 2, '')

flags.DEFINE_float('alpha', 1.15, '')
flags.DEFINE_float('lambd', 10.6, '')

flags.DEFINE_float('lr', 1e-4, '')
flags.DEFINE_float('beta', 0.99, '')
flags.DEFINE_integer('num_epochs', 20000, '')
flags.DEFINE_integer('batch_size', 256, 'batch size for the main network')

flags.DEFINE_multi_integer('layer_width_generator', [10, 5], '')
flags.DEFINE_multi_integer('layers_width', [15, 10, 10], 'The width of the layers in main network')
flags.DEFINE_string('nonlin','tanh', '')
flags.DEFINE_string('nonlin_dataset','Relu', '')
flags.DEFINE_integer('num_of_save_steps', 50,'')
flags.DEFINE_integer('steps_per_epoch', 20000,'')

mlflow.tensorflow.autolog(1)
class LoggerHistory(tf.keras.callbacks.Callback):
    def __init__(self, val_data, train_data, steps, file_name, checkpoint_path):
            super().__init__()
            self.validation_data = val_data
            self.steps = steps
            self.train_data = train_data
            self.step_counter = 0
            self.file_name = file_name
            self.checkpoint_path = checkpoint_path
            list_of_elem = ['step', 'acc', 'loss', 'val_loss', 'val_acc']
            with open(self.file_name, 'w+', newline='') as write_obj:
                # Create a writer object from csv module
                csv_writer = writer(write_obj)
                # Add contents of list as last row in the csv file
                csv_writer.writerow(list_of_elem)

    def on_epoch_begin(self, epoch, logs=None):
        self.train_loss = []
        self.train_acc = []

    def on_batch_end(self, batch, logs={}):
        self.train_loss.append(logs.get('loss'))
        self.train_acc.append(logs.get('accuracy'))
        if  self.step_counter in self.steps:
            val_loss, val_acc = self.model.evaluate(self.validation_data, steps =1000)
            loss, acc = self.model.evaluate(self.train_data, steps = 1000)
            list_of_elem = [self.step_counter, acc, loss,val_loss, val_acc ]
            with open(self.file_name, 'a+', newline='') as write_obj:
                # Create a writer object from csv module
                csv_writer = writer(write_obj)
                # Add contents of list as last row in the csv file
                csv_writer.writerow(list_of_elem)
            self.model.save(self.checkpoint_path.format(self.step_counter))
        self.step_counter+=1

def train(train_ds, test_ds, lr, beta, num_epochs, input_shape, y_dim, nonlin, layers_width, steps_per_epoch=1):
    """Train the model and measure the information."""
    model = bulid_model(input_shape=input_shape, y_dim=y_dim, nonlin=nonlin, layers_width=layers_width)
    optimizer = tf.keras.optimizers.SGD(lr, beta)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=['accuracy'])

    checkpoint_path = mlflow.get_artifact_uri()+"/model/checkpoints/{}_cp"
    logdir = mlflow.get_artifact_uri()+ "/model"
    # Create a callback that saves the model's weights
    steps = np.unique(np.logspace(0, np.log10(num_epochs *steps_per_epoch), FLAGS.num_of_save_steps, dtype =np.int  ))
    history = LoggerHistory(steps=steps, val_data=test_ds,train_data=train_ds,
                            file_name=mlflow.get_artifact_uri()+'/c_looger.csv', checkpoint_path=checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_best_only=False,
                                                     save_weights_only=False,
                                                     verbose=0)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    csv_logger = tf.keras.callbacks.CSVLogger(
        logdir + '/logger.csv', separator=',', append=False
    )
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