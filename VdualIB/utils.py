"""Utils for the VdualIB"""
from csv import writer
from os import environ

import mlflow
import pandas as pd
import tensorflow as tf


def save_pickle(file_name, list_of_elem):
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


class LoggerTrain(tf.keras.callbacks.Callback):
    """Save loss and accuracy to csv and model checkpoints"""

    def __init__(self, file_name, checkpoint_path):
        super().__init__()
        print(file_name, checkpoint_path)
        self.step_countLoggerTrainer = 0
        self.file_name = file_name
        self.checkpoint_path = checkpoint_path
        self.df = pd.DataFrame()

    def _log_epoch_metrics(self, epoch, logs):
        """Writes epoch metrics out as scalar summaries.
        Arguments:
          epoch: Int. The global step to use for TensorBoard.
          logs: Dict. Keys are scalar summary names, values are scalars.
        """
        if not logs:
            return

        train_logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        train_logs['mode'] = 'train'
        train_logs['step'] = epoch
        val_logs = {k[4:]: v for k, v in logs.items() if k.startswith('val_')}  # Remove 'val_' prefix.
        val_logs['mode'] = 'val'
        val_logs['step'] = epoch
        if train_logs:
            self.df = self.df.append(train_logs, ignore_index=True)
        if val_logs:
            self.df = self.df.append(val_logs, ignore_index=True)
        self.df.to_csv(self.file_name)

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_batch_end(self, batch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs=None):
        """Runs metrics and histogram summaries at epoch end."""
        self._log_epoch_metrics(epoch, logs)
        self.model.save_weights(self.checkpoint_path.format('_1'))


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d


def insert_keys(FLAGS):
    dict = dotdict()

    for key in FLAGS.flag_values_dict():
        # try:
        if True:
            val_env = environ.get(key)
            value = FLAGS.get_flag_value(key, None)
            if val_env is not None:
                if type(value) == list:
                    value = [val_env]
                elif type(value) == bool:
                    value = val_env == 'True'
                elif type(value) != list:
                    value = type(value)(val_env)
            print(key, value)
            dict[key] = value
        # except:
        #    continue
    if dict.run_model == 'det_wide_resnet':
        dict.metrices = ['hyz', 'hyhatz', 'izyhat']
    run_id = mlflow.active_run().info.run_id
    dict['run_id'] = run_id
    return dict
