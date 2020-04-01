"""This file load the trained networks and calculate the diffrneet information quantities  """

from __future__ import absolute_import, division, print_function, unicode_literals
import logging, os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import mlflow
import pathlib
from functools import  partial
import mlflow.tensorflow
import tensorflow as tf
from absl import flags
import time
import numpy as np
from absl import app
import glob
import pandas as pd
from datasets import create_dataset, create_dataset_np, get_data
from utils import  load_matrices2, store_data2, log_summary2, process_list
from estimators.information_estimators import get_information_all_layers_clusterd, get_nonlinear_information
from estimators.MINE import get_information_all_layers_MINE
from estimators.binning_MI import  get_information_bins_estimators2
from estimators.dual_ib import beta_func
FLAGS = flags.FLAGS
flags.DEFINE_integer('num_of_epochs_inf_labels', 200, '')
flags.DEFINE_string('csv_path','data.csv', '')
#flags.DEFINE_multi_integer('num_of_clusters', [3,3,3,3], '')
flags.DEFINE_integer('max_clusters', 1000, 'Maximum number of clusters')
flags.DEFINE_integer('min_clusters', 2, 'Minimum number of clusters')
flags.DEFINE_integer('num_of_clusters_run', 5, 'Number of diffrenet runs')
flags.DEFINE_integer('num_of_samples', 1, 'For the NCE estimator')
flags.DEFINE_multi_float('binsize', [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1., 1.5],  'The size of the bins')

#ffce9455bd8048bb9e964d6e98cd0118
flags.DEFINE_string('run_id', '30cefabbb5394a21b97a9cd36ae193f4', 'The id of the run that you want to load')
flags.DEFINE_string('experiment_id', '2', 'the experiment_id to load')

flags.DEFINE_multi_float('noisevar', [0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.3, 0.5,0.8, 1], '')
flags.DEFINE_float('lr_labels', 5e-2, '')
flags.DEFINE_integer('num_test', 1300, 'Number of test examples')
flags.DEFINE_integer('mine_epochs', 1500 , "")
flags.DEFINE_integer('num_of_steps', 200, 'The number of steps to calculate')
flags.DEFINE_multi_enum('information_measures', [ 'CLUSTERED'],
                        ['BINS', 'NONLINEAR', 'MINE', 'CLUSTERED', 'DUAL_IB'], 'Which information measure to calculate')

def get_informations(batch_test, model, py, py_x, xs, A, lambd, px, information_measures =None, num_of_clusters=None):

    information_bins =  linear_information= information_MINE= information_clustered= information_dual_ib =[]
    ts = [tf.keras.Model(model.inputs, model.layers[layer_index].output)(batch_test) for layer_index in range(len(model.layers))]
    if 'BINS' in information_measures:
        information_bins = get_information_bins_estimators2(batch_test=batch_test, ts=ts, binsize=FLAGS.binsize)
    if 'NONLINEAR' in information_measures:
        linear_information = get_nonlinear_information(ts, batch_test, py.entropy(), FLAGS.num_of_epochs_inf_labels,
                                                   FLAGS.lr_labels, FLAGS.noisevar)
    if 'MINE' in information_measures:
        information_MINE = get_information_all_layers_MINE(model=model, x_test=batch_test[0], y_test=batch_test[1],
                                                       batch_size=256, epochs=1500)
    if 'CLUSTERED' in information_measures:
        information_clustered, information_dual_ib = get_information_all_layers_clusterd(
            data_all =ts, num_of_clusters=num_of_clusters, py_x=py_x, py=py,
         px=px,  calc_dual='DUAL_IB' in information_measures, A=A, lambd=lambd)
    return information_bins, linear_information, information_MINE, information_clustered, information_dual_ib

#@tf.function
def process_steps_inner(batch_test, model, num_of_clusters, py, py_x, xs, px, A, lambd):
    information_bins, linear_information, information_MINE, information_clustered, information_dual_ib = \
        get_informations(batch_test, model, py, py_x, xs, A, lambd, px,
                         information_measures=FLAGS.information_measures, num_of_clusters=num_of_clusters)
    return linear_information, information_MINE, information_clustered, information_dual_ib, information_bins


def evulate_model(model_path, batch_train, batch_test):
    model = tf.keras.models.load_model(model_path)
    model.save_weights(os.path.join(model_path, 'weights'))
    x, y = batch_train
    loss_value = model.evaluate(x, y, verbose = 0)
    # get all the information related to the test data
    test_loss_val = model.evaluate(batch_test[0], batch_test[1], verbose = 0)
    return loss_value, test_loss_val, model



#@tf.function
def proess_step(i, matrices, df_tenosr,  dirs ,num_of_clusters,
                batch_test, py, py_x, xs, px, A, lambd,
                train_losses, test_losses, models):
    tf.print('Step - ', i)
    csv_path = os.path.join(mlflow.get_artifact_uri(), 'data_{}.csv'.format(i))
    model_path = dirs[i]
    model = models[i]
    test_loss_val = test_losses[i]
    loss_value = train_losses[i]
    steps = [ext_key(k) for k in dirs]
    with tf.device('/CPU:0'):
        linear_information,  information_MINE, information_clustered,  information_dual_ib,  information_bins =\
            process_steps_inner(batch_test, model, num_of_clusters, py, py_x, xs, px, A, lambd)
    df_list, names_list = store_data2(matrices=matrices, step=i, loss_value=loss_value, test_loss_val=test_loss_val,
                                      nonlinear_information=linear_information, information_MINE=information_MINE,
                                      information_clustered=information_clustered,
                                      information_dual_ib=information_dual_ib, information_bins=information_bins)
    df_tenosr[i] = df_list
    df = pd.DataFrame(data=np.array(df_tenosr), columns=names_list)

    df['step'] = steps
    try:
        df.to_csv(csv_path)
    except:
        pass
    return i+1, matrices, df_tenosr


def get_losses(dirs, test_ds, train_ds):
    train_losses, test_losses, models = [], [], []

    for batch_train in train_ds.take(tf.constant(1, dtype=tf.int64)):
        break
    for batch_test in test_ds.take(1):
        break
    for i in range(len(dirs)):
        loss_value, test_loss_val, model = evulate_model(dirs[i], batch_test, batch_train)
        models.append(model)
        train_losses.append(loss_value)
        test_losses.append(test_loss_val)
    return models, train_losses, test_losses, batch_test, batch_train

ext_key = lambda x: int(x.split('/')[-1].split('_')[0])


def main(argv):

    with mlflow.start_run():

        mlflow.log_param("max_clusters", FLAGS.max_clusters)
        mlflow.log_param("min_clusters", FLAGS.min_clusters)
        mlflow.log_param("num_of_clusters_run", FLAGS.num_of_clusters_run)
        mlflow.log_param("binsize", FLAGS.binsize)
        mlflow.log_param("run_id", FLAGS.run_id)
        mlflow.log_param("experiment_id", FLAGS.experiment_id)
        mlflow.log_param("noisevar", FLAGS.noisevar)
        mlflow.log_param("num_test", FLAGS.num_test)
        mlflow.log_param("mine_epochs", FLAGS.mine_epochs)

        run = mlflow.get_run(run_id=FLAGS.run_id)
        params = run.data.params
        new_base_line = mlflow.get_artifact_uri()
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        model_path = './mlruns/{}/{}/artifacts/model/checkpoints/*'.format(FLAGS.experiment_id, FLAGS.run_id)
        layer_widths = process_list(params['layer_width_generator'])
        num_of_layers = len(process_list(params['layers_width']))+1
        if FLAGS.max_clusters > 0:
            num_of_clusters = [np.unique(np.logspace(np.log10(FLAGS.min_clusters),
                                                     np.log10(FLAGS.max_clusters),
                                                     FLAGS.num_of_clusters_run, dtype=np.int))
                               for i in range(num_of_layers)]
        else:
            num_of_clusters =[[FLAGS.num_of_clusters[i]] for i in range(FLAGS.num_of_clusters)]
        path = os.path.join(mlflow.get_artifact_uri(), FLAGS.csv_path)
        print ('PATH', path, flush = True)
        try:
            if not os.path.exists(mlflow.get_artifact_uri()):
                pathlib.Path(mlflow.get_artifact_uri()).mkdir(parents=True, exist_ok=True)
            logger.addHandler(logging.FileHandler(os.path.join(new_base_line, 'log_file')))

        except:
            pass
        strategy = tf.distribute.MirroredStrategy()
        #with strategy.scope():
        if True:
            train_ds, test_ds, py, py_x, xs, px, A, lambd = create_dataset(
                                                                             int(params['num_train']),
                                                                           int(FLAGS.num_test),
                                                                           int(params['x_dim']), layer_widths,
                                                                           params['nonlin_dataset'],
                                                                           batch_size=int(FLAGS.num_test),
                                                                           lambd_factor=float(params['lambd']),
                                                                           alpha=float(params['alpha']))
            dirs = glob.glob(model_path)
            #Go over all the directories in the path (epochs)
            dirs.sort(key=ext_key)
            #dirs = dirs[:2]
            model = tf.keras.models.load_model(dirs[0])
            matrices = load_matrices2(num_of_layers=num_of_layers, num_of_clusters=num_of_clusters, num_of_epochs=len(dirs),
                                      num_of_bins = len(FLAGS.binsize), num_of_noises=len(FLAGS.noisevar))
            df_list = np.zeros((len(dirs),
                                #bins
                                2*len(model.layers)* len(FLAGS.binsize)+

                                #mine
                                2*len(model.layers)+
                                #nonlinear
                                2 * len(model.layers)*len(FLAGS.noisevar)+
                                #clustered + dual
                                len(num_of_clusters[0])*4*len(model.layers)+
                               #losses
                               4) , dtype = np.float64)
            def cond(i, *args):
                return i < len(dirs)
            t = time.time()
            models, train_losses, test_losses, batch_test, batch_train = get_losses(dirs, test_ds, train_ds)

            part_process = partial(proess_step, dirs=dirs,
                                   num_of_clusters=num_of_clusters,
                                   batch_test=batch_test, py= py, py_x = py_x, xs=xs, px=px, A=A,
                                   lambd=lambd, train_losses=train_losses, test_losses=test_losses, models=models
                                   )
            _,_, _output_array = tf.while_loop(
                cond, part_process, [0,matrices, df_list],
                parallel_iterations=100,  swap_memory=True)
            print ('tttime', time.time()-t)

if __name__ == "__main__":
        app.run(main)