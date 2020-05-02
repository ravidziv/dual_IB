"""This file load the trained networks and calculate the diffrneet information quantities  """

from __future__ import absolute_import, division, print_function, unicode_literals
import logging, os
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import mlflow
import pathlib
from functools import  partial
import mlflow.tensorflow
import tensorflow as tf
#tf.keras.backend.set_floatx('float64')
from absl import flags
import time
import numpy as np
from absl import app
from sklearn import mixture
import matplotlib.pyplot as plt
import glob
import pandas as pd
from datasets import create_dataset
from utils import  load_matrices2, store_data2, process_list
from estimators.information_estimators import get_information_all_layers_clusterd, get_nonlinear_information
from estimators.MINE import information_mine
from estimators.binning_MI import  get_information_bins_estimators2
from estimators.dual_ib import beta_func
FLAGS = flags.FLAGS
flags.DEFINE_integer('num_of_epochs_inf_labels', 100, '')
flags.DEFINE_string('csv_path','data.csv', '')
flags.DEFINE_integer('max_clusters', 100, 'Maximum number of clusters')
flags.DEFINE_integer('min_clusters', 5, 'Minimum number of clusters')
flags.DEFINE_integer('num_of_clusters_run', -1, 'Number of diffrenet runs')
flags.DEFINE_integer('num_of_samples', 1, 'For the NCE estimator')
flags.DEFINE_multi_float('binsize', [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.6, 1.,1.5, 2, 1.5, 4, 6, 8],  'The size of the bins')
#Relu
#flags.DEFINE_string('run_id', '93973014b78f4e7abcbac843d3571161', 'The id of the run that you want to load')
flags.DEFINE_string('run_id', '56337557390340e798847276e3ae81d0', 'The id of the run that you want to load')
flags.DEFINE_string('experiment_id', '0', 'the experiment_id to load')
flags.DEFINE_multi_float('noisevar', [1e-2, 1e-1, 5e-1, 1e0, 1e1], '')
flags.DEFINE_multi_float('dual_betas',
                         [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,
                          1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5, 3, 4], 'betas values for dual ib')


flags.DEFINE_float('lr_labels', 5e-4, '')
flags.DEFINE_integer('num_test', 100, 'Number of test examples')
#flags.DEFINE_integer('num_train', 50000, 'Number of test examples')
flags.DEFINE_integer('mine_epochs', 5 , "The number of epochs for the mine estimator")
flags.DEFINE_integer('mine_num_samples', 10 , "The number of samples for the mine estimator")
flags.DEFINE_integer('batch_size_mine', 5000 , "The batch size for the mine estimator")
flags.DEFINE_multi_enum('information_measures', ['CLUSTERED', 'DUAL_IB'],
                        ['BINS', 'NONLINEAR', 'MINE', 'CLUSTERED', 'DUAL_IB'], 'Which information measure to calculate')



def get_informations(model, py, py_x, xs, A, lambd, px, information_measures =None,
                     num_of_clusters=None, clfs=None, paths=None, ixy= 0):

    information_bins =  linear_information= inf_mine= information_clustered= information_dual_ib = normalized_information_KDE =[]
    ts_layers = [tf.keras.Model(model.inputs, model.layers[layer_index].output)(xs) for layer_index in range(len(model.layers))]
    t = time.time()
    batch_data = [xs, py_x.probs]
    if 'BINS' in information_measures:
        information_bins = get_information_bins_estimators2(batch_test=batch_data, ts=ts_layers, binsize=FLAGS.binsize)
        print ('Bins', time.time()-t)
        t = time.time()

    if 'NONLINEAR' in information_measures:

        linear_information, normalized_information_KDE = get_nonlinear_information(ts_layers, batch_data, py.entropy(), FLAGS.num_of_epochs_inf_labels,
                                                                                   lr_labels = FLAGS.lr_labels, noisevar=FLAGS.noisevar,
                                                                                   py_probs=py.probs, py_x=py_x, px_probs=px.probs,
                                                                                   paths=paths, layer_width=[30, 20, 10])
        print('NONLINEAR', time.time() - t)
        t = time.time()
    if 'MINE' in information_measures:
        inf_mine = information_mine(px=px, xs=xs, ts_layers = ts_layers, py_x=py_x,
                                    ixy=ixy, num_of_samples=FLAGS.mine_num_samples,
                                    batch_size=FLAGS.batch_size_mine, epochs=FLAGS.mine_epochs)
        print('Mine', time.time() - t)
        t = time.time()
    if 'CLUSTERED' in information_measures:
        information_clustered, information_dual_ib = get_information_all_layers_clusterd(
            clustered_data = [],clfs=clfs, targets = py_x.probs, num_of_epochs_inf_labels=FLAGS.num_of_epochs_inf_labels,
            data_all =ts_layers, train_data_all=ts_layers, num_of_clusters=num_of_clusters, py_x=py_x, py=py,
            px=px, calc_dual='DUAL_IB' in information_measures, A=A, lambd=lambd, betas=FLAGS.dual_betas)
        print('CLUSTERED', time.time() - t)
    return information_bins, linear_information, inf_mine, information_clustered, information_dual_ib, normalized_information_KDE



def evaluate_model(model_path, batch_train, batch_test):
    model = tf.keras.models.load_model(model_path)
    x, y = batch_train
    #return [0,0], [0,0], model
    loss_value = model.evaluate(x, y, verbose = 0)
    test_loss_val = model.evaluate(batch_test[0], batch_test[1], verbose = 0)
    return loss_value, test_loss_val, model



def proess_step(i, matrices, df_tenosr,  clfs, dirs ,num_of_clusters,
                 py, py_x, xs, px, A, lambd,
                train_losses, test_losses, models, paths, ixy):
    tf.print('Step - ', i)
    csv_path = os.path.join(mlflow.get_artifact_uri(), 'data_{}.csv'.format(i))
    model_path = dirs[i]
    model = models[i]
    test_loss_val = test_losses[i]
    loss_value = train_losses[i]
    steps = [ext_key(k) for k in dirs]
    with tf.device('/CPU:0'):
        information_bins, linear_information, information_MINE, information_clustered, information_dual_ib, normalized_information_KDE = \
            get_informations(model=model, num_of_clusters=num_of_clusters, py=py, py_x=py_x, xs=xs, px=px, A=A,
                             lambd=lambd, clfs=clfs,
                                paths=paths, ixy=ixy,  information_measures=FLAGS.information_measures)
    df_list, names_list = store_data2(matrices=matrices, step=i, loss_value=loss_value, test_loss_val=test_loss_val,
                                      nonlinear_information=linear_information, information_MINE=information_MINE,
                                      information_clustered=information_clustered,
                                      normalized_information_KDE=normalized_information_KDE,
                                      information_dual_ib=information_dual_ib, information_bins=information_bins
                                      )
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
    for batch_train, batch_test in zip(train_ds.take(1), test_ds.take(1)):
        for i in range(len(dirs)):
            print (i)
            loss_value, test_loss_val, model = evaluate_model(dirs[i], batch_test, batch_train)
            models.append(model)
            train_losses.append(loss_value)
            test_losses.append(test_loss_val)
    return models, train_losses, test_losses

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
        model_path = './mlruns/{}/{}/artifacts/model1/checkpoints/*'.format(FLAGS.experiment_id, FLAGS.run_id)
        layer_widths = process_list(params['layer_width_generator'])
        num_of_layers = len(process_list(params['layers_width']))+1
        if FLAGS.num_of_clusters_run > 0:
            num_of_clusters = [np.unique(np.logspace(np.log10(FLAGS.min_clusters),
                                                     np.log10(FLAGS.max_clusters),
                                                     FLAGS.num_of_clusters_run, dtype=np.int))
                               for i in range(num_of_layers)]
        else:
            num_of_clusters =[[FLAGS.max_clusters] for i in range(num_of_layers)]
        path = os.path.join(mlflow.get_artifact_uri(), FLAGS.csv_path)
        print ('PATH', path, flush = True)
        try:
            if not os.path.exists(mlflow.get_artifact_uri()):
                pathlib.Path(mlflow.get_artifact_uri()).mkdir(parents=True, exist_ok=True)
            logger.addHandler(logging.FileHandler(os.path.join(new_base_line, 'log_file')))

        except:
            pass
        strategy = tf.distribute.MirroredStrategy()
        train_ds, test_ds, py, py_x, xs, px, A, lambd, ixy = create_dataset(int(FLAGS.num_test),
                                                                            int(FLAGS.num_test),
                                                                            int(params['x_dim']), layer_widths,
                                                                            params['nonlin_dataset'],
                                                                            batch_size=int(FLAGS.num_test),
                                                                            train_batch_size=int(FLAGS.num_test),
                                                                            lambd_factor=float(params['lambd']),
                                                                            alpha=float(params['alpha']))
        dirs = glob.glob(model_path)
        #Go over all the directories in the path (epochs)
        dirs.sort(key=ext_key)
        dirs = dirs[-3:]
        mlflow.log_param('model_path', model_path)
        model = tf.keras.models.load_model(dirs[0])
        matrices = load_matrices2(num_of_layers=num_of_layers, num_of_clusters=num_of_clusters, num_of_epochs=len(dirs),
                                  num_of_bins = len(FLAGS.binsize), num_of_noises=len(FLAGS.noisevar))
        df_list = np.zeros((len(dirs),
                            #bins
                            2 * len(model.layers) * len(FLAGS.binsize) +
                            #mine
                            2 * len(model.layers) +
                            #nonlinear
                            2 * len(model.layers) * len(FLAGS.noisevar) +
                            2 * len(model.layers) * len(FLAGS.noisevar) +
                            # clustered +
                            len(num_of_clusters[0]) * 2 * len(model.layers) +
                            # dual
                            len(num_of_clusters[0]) * 1 * len(FLAGS.dual_betas) * len(model.layers) +

                            # losses
                            4) , dtype = np.float64)
        def cond(i, *args):
            return i < len(dirs)
        t = time.time()
        models, train_losses, test_losses = get_losses(dirs, test_ds, train_ds)
        clfs = [[mixture.GaussianMixture(n_components=num_of_clusters[i][j], warm_start=True, verbose=0,
                                         covariance_type='spherical', max_iter=200, tol=1e-4, reg_covar=1e-4)
                 for j in range(len(num_of_clusters[i]))] for
                 i in range(len(num_of_clusters))]
        #path = os.path.join(new_base_line,
        path = '{}{}'
        paths = [[path.format(i,j) for j in range(len(FLAGS.noisevar))]for i in range(num_of_layers+1)]
        part_process = partial(proess_step, dirs=dirs,
                               num_of_clusters=num_of_clusters,clfs=clfs, py= py, py_x = py_x, xs=xs, px=px, A=A,
                               lambd=lambd, train_losses=train_losses, test_losses=test_losses, models=models, paths=paths,
                               ixy=ixy
                               )
        _,_, _output_array = tf.while_loop(
            cond, part_process, [0,matrices, df_list],
            parallel_iterations=100,  swap_memory=True)

        print ('tttime', time.time()-t)

if __name__ == "__main__":
        app.run(main)