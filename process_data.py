"""This file load the trained networks and calculate the diffrneet information quantities  """

from __future__ import absolute_import, division, print_function, unicode_literals
import logging, os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from absl import flags
from absl import app
import glob
import pandas as pd
from datasets import create_dataset
from utils import store_data, log_summary, load_matrices
from information_estimators import get_information_all_layers_clusterd, get_nonlinear_information
from MINE import get_information_all_layers_MINE
from binning_MI import  get_information_bins_estimators2
from dual_ib import get_information_dual_all_layers, beta_func
FLAGS = flags.FLAGS
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.FileHandler('log_file'))

flags.DEFINE_integer('num_of_epochs_inf_labels', 2, '')
flags.DEFINE_string('csv_path','data.csv', '')
flags.DEFINE_multi_integer('num_of_clusters', [3, 3, 3, 3], 'The width of the layers in the random network')
flags.DEFINE_integer('num_of_samples', 2, '')
flags.DEFINE_float('binsize', 0.1, 'The size of the bins')
flags.DEFINE_string('run_id', '7dcef91d89654b40a56e58114bde1ca2', 'The id of the run that you want to load')
flags.DEFINE_float('noisevar', 1e-1, '')
flags.DEFINE_float('lr_labels', 5e-2, '')

def process_list(params_val):
    return [int(s) for s in params_val.split('[')[1].split(']')[0].split(',')]


def get_informations(batch_test, model, py, py_x, xs, A, lambd, px):
    information_bins = get_information_bins_estimators2(batch_test, model, binsize=FLAGS.binsize)
    linear_information = get_nonlinear_information(model, batch_test, py.entropy(), FLAGS.num_of_epochs_inf_labels,
                                                   FLAGS.lr_labels, FLAGS.noisevar)
    information_MINE = get_information_all_layers_MINE(model=model, x_test=batch_test[0], y_test=batch_test[1],
                                                       batch_size=500)
    information_clustered = get_information_all_layers_clusterd(
        model=model, x_test=batch_test[0], num_of_clusters=FLAGS.num_of_clusters, py_x=py_x, xs=xs, py=py,
        num_of_samples=FLAGS.num_of_samples)
    information_dual_ib = get_information_dual_all_layers(model=model, num_of_clusters=FLAGS.num_of_clusters, xs=xs,
                                                          A=A, lambd=lambd, px=px, py=py, beta_func=beta_func)

    return information_bins, linear_information, information_MINE, information_clustered, information_dual_ib


def main(argv):
    with mlflow.start_run():
        run = mlflow.get_run(run_id=FLAGS.run_id)
        params = run.data.params
        model_path = './mlruns/0/{}/artifacts/model/checkpoints/*'.format(FLAGS.run_id)
        layer_widths = process_list(params['layer_widths'])
        num_of_layers = len(process_list(params['layers_width']))+1
        matrices = load_matrices(num_of_layers=num_of_layers)
        summary_path = mlflow.get_artifact_uri() + "/tensorboard_logs_n"
        path = mlflow.get_artifact_uri() + "/model/data/{}".format(FLAGS.csv_path)
        summary_writer = tf.summary.create_file_writer(summary_path)
        train_ds, test_ds, py, py_x, xs, px, A, lambd = create_dataset(int(params['num_train']),
                                                                       int(params['num_test']),
                                                                       int(params['x_dim']), layer_widths,
                                                                       params['nonlin_dataset'],
                                                                       batch_size=int(params['batch_size']),
                                                                       lambd_factor=float(params['lambd']),
                                                                       alpha=float(params['alpha']))
        dirs = glob.glob(model_path)
        dfs  =[]
        #Go over all the directories in the path (epochs)
        for model_path in dirs:
            epoch = int(model_path.split('/')[-1].split('_')[0])
            model = tf.keras.models.load_model(model_path)
            #get the train loss
            for batch_train in train_ds.take(1):
                loss_value = model.evaluate(batch_train[0],batch_train[1])
            #get all the information related to the test data
            for batch_test in test_ds.take(1):
                test_loss_val = model.evaluate(batch_test[0], batch_test[1])
                information_bins, linear_information, information_MINE, information_clustered, information_dual_ib =\
                    get_informations(batch_test, model, py, py_x, xs, A, lambd, px)
                store_data(matrices, loss_value, test_loss_val, linear_information, information_MINE, information_clustered,
                           information_dual_ib, information_bins)
                df = log_summary(summary_writer, epoch, matrices, logger)
                dfs.append(df)
        dfs = pd.concat(dfs, axis=0, sort=True)
        dfs.to_csv(path)


if __name__ == "__main__":
        app.run(main)