"""Implement a  family of variational models - ceb, vib and the dual ib """
try:
    import cdsw
    found_cdsw = True
except ImportError:
    found_cdsw = False
use_cdsw = False

import os
import pickle
import sys
from os import environ
sys.path.insert(0, "/home/cdsw/")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import tensorflow as tf
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import mlflow
from utils import LoggerTrain, dotdict
from absl import flags
from absl import app
import tensorflow_probability as tfp
from functools import partial
# from VdualIB.pgd_t2 import get_adversarial_acc_metric
from network_utils import build_covnet, build_densnet

from VdualIB.schedulers import *
from VdualIB.models.models import BasePrior, BaseDecoder, BasedEncoder, BZYPrior, BaseLabelsModel, build_default_FC_net, \
    build_default_cov_net
from VdualIB.models.wide_resnet import wide_residual_network
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from VdualIB.variational_network import loss_func_ib, loss_func_dual_ib, VariationalNetwork, loss_func_combined
from datasets import load_cifar_data, load_mnist_data, load_fasion_mnist_data, load_cifar100_data

# from cleverhans.attacks import FastGradientMethod
# from cleverhans.utils_keras import KerasModelWrapper

tfkl = tf.keras.layers
dfd = tfp.distributions
FLAGS = flags.FLAGS
import mlflow.tensorflow

flags.DEFINE_integer('end_anneling', 4000, 'In which step to stop annealing beta')

flags.DEFINE_integer('z_dim', 320, 'The dimension of the hidden space')
flags.DEFINE_integer('num_of_labels', 10, 'The number of labels')
flags.DEFINE_integer('h_dim', 10, 'the dimension of the hidden layers of the encoder')
flags.DEFINE_integer('batch_size', 64, 'For training')
flags.DEFINE_integer('num_of_epochs', 10000, 'For training')
flags.DEFINE_string('activation', 'relu', 'Activation of the encoder layers')
flags.DEFINE_string('log_file_path', 'w_resnet', 'Where to save the network tensorboard')
flags.DEFINE_string('dataset', 'cifar', 'Which dataset to load (mnist, fasion_mnist, cifar, cifar100)')
flags.DEFINE_string('encoder_type', 'wide_resnet', 'The encoder of the network - [wide_resnet, efficientNetB0, '
                                                   'covnet, FC]')
flags.DEFINE_string('opt', 'adam', 'Which learning method to train with')

flags.DEFINE_multi_enum('run_model', 'ceb',
                        ['vib', 'ceb', 'dual_ib', 'det_wide_resnet', 'combined', 'det_covnet', 'det_fc'],
                        'Which model to run')
flags.DEFINE_float('initial_lr', 1e-4, 'The lr for the train')
flags.DEFINE_float('momentum', 0.9, 'The momentum of the SGD')
flags.DEFINE_float('log_beta', 10000., 'log_beta value for the loss function')
flags.DEFINE_float('gamma', 0., 'gamma value for the loss function')

flags.DEFINE_float('weights_decay', 0.0005, 'The weights decay training')
flags.DEFINE_float('labels_noise', 1e-2, 'The noise for calculating the kl of the labels in the dual ib')
flags.DEFINE_string('label_noise_type', 'confusion_matrix_noise',
                    ['confusion_matrix_noise, gaussian_noise', 'smooth_noise', 'pre_defined_model'])
flags.DEFINE_integer('depth', 28, 'the depth of the wide resent newtork')
flags.DEFINE_integer('wide', 10, 'the wide of the wide resent newtork')
flags.DEFINE_integer('num_of_train', -1, 'The number of train examples for the training, -1 for all')
flags.DEFINE_integer('gpu', 0, 'gpu index')
flags.DEFINE_multi_string('metrices',
                          ['hzy', 'hzx', 'hyz', 'hyhatz', 'ixt', 'izy', 'izyhat', 'hyz_noisy', 'total_loss', 'beta',
                           'gamma'],
                          'The metrices to save')
flags.DEFINE_bool("noisy_learning", True, 'Train ceb/vib models with noisy labels')
flags.DEFINE_bool("use_logprob_func", False, 'Sample label noise from gaussian or empirical mle')
flags.DEFINE_bool("train_model", True, 'train or test model')
flags.DEFINE_string('pre_model_path', 'e226f66bbee1423e911ea63841c6d80e', '')
flags.DEFINE_bool("with_batchnorm", True, 'For adve attacks we need to remove the batchnorm layers')
flags.DEFINE_string('version', '1', 'The version of the file')


def insert_keys():
    dict = dotdict()

    for key in FLAGS.flag_values_dict():
        try:
            val_env = environ.get(key)
            value = FLAGS.get_flag_value(key, None)
            if val_env is not None:
                if type(value) == list:
                    value = [val_env]
                elif type(value) == bool:
                    value = val_env == 'True'
                elif type(value) != list:
                    value = type(value)(val_env)
            mlflow.log_param(key, value)
            print(key, value)
            if found_cdsw:
                cdsw.track_metric(key, value)
            dict[key] = value
        except:
            continue
    if dict.run_model[0] == 'det_wide_resnet':
        dict.metrices = ['hyz', 'hyhatz', 'izyhat']
    run_id = mlflow.active_run().info.run_id
    dict['run_id'] = run_id
    return dict


def get_dataset(dict):
    if dict.dataset == 'mnist':
        ds_train, ds_test, steps_per_epoch = load_mnist_data(batch_size=dict.batch_size,
                                                             num_epochs=dict.num_of_epochs)
    elif dict.dataset == 'fasion_mnist':
        ds_train, ds_test, steps_per_epoch, steps_per_epoch_validation, confusion_matrix = load_fasion_mnist_data(
            batch_size=dict.batch_size, num_of_train=dict.num_of_train,
            num_epochs=dict.num_of_epochs)
    elif dict.dataset == 'cifar100':
        ds_train, ds_test, steps_per_epoch, steps_per_epoch_validation, confusion_matrix = load_cifar100_data(
            batch_size=dict.batch_size, num_of_train=dict.num_of_train,
            num_epochs=dict.num_of_epochs)
    else:
        ds_train, ds_test, steps_per_epoch, steps_per_epoch_validation, confusion_matrix = load_cifar_data(
            num_class=dict.num_of_labels, batch_size=dict.batch_size, num_of_train=dict.num_of_train)
    return ds_train, ds_test, steps_per_epoch, steps_per_epoch_validation, confusion_matrix


def get_vmodel(img_input, confusion_matrix, dict, log_file_path_name):
    # Create encoder, decoder and the model
    if dict.run_model[0] == 'ceb' or dict.run_model[0] == 'vib':
        loss_func_inner = partial(loss_func_ib, noisy_learning=dict.noisy_learning)
        if dict.run_model == 'ceb':
            prior = BZYPrior(z_dims=dict.z_dim, num_classes=dict.num_of_labels)
        else:
            prior = BasePrior(z_dims=dict.z_dim)

    elif dict.run_model[0] == 'dual_ib':
        loss_func_inner = loss_func_dual_ib
        prior = BZYPrior(z_dims=dict.z_dim, num_classes=dict.num_of_labels)
    elif dict.run_model[0] == 'combined':
        loss_func_inner = partial(loss_func_combined, noisy_learning=dict.noisy_learning)
        prior = BZYPrior(z_dims=dict.z_dim)
    labels_dist = BaseLabelsModel(confusion_matrix=confusion_matrix, num_claases=dict.num_of_labels)
    if dict.dataset == 'mnist':
        encoder = BasedEncoder(z_dim=dict.z_dim, h_dim=dict.h_dim, activation=dict.activation,
                               layer_input_shape=(28, 28, 1))
        beta_sched = beta_sched_mnist
        scheduler = scheduler_mnist
    else:
        beta_sched = partial(lerp, start_step=0, end_step=dict.end_anneling, start_val=100,
                             end_val=dict.log_beta)
        gamma_sched = partial(lerp, start_step=0, end_step=dict.end_anneling, start_val=100,
                              end_val=dict.gamma)
        scheduler = scheduler_cifar

        if dict.encoder_type == 'wide_resnet':
            output = wide_residual_network(img_input, dict.num_of_labels, dict.depth, dict.wide,
                                           dict.weights_decay,
                                           is_cifar=(dict.dataset == 'cifar' or dict.dataset == 'cifar100'),
                                           with_batchnorm=dict.with_batchnorm)
            net = Model(img_input, output)
        if dict.encoder_type == 'efficientNetB0':
            net = tf.keras.applications.EfficientNetB0(
                include_top=False, weights=None, input_tensor=img_input, input_shape=None,
                pooling=None, classes=10)
        if dict.encoder_type == 'covnet':
            net = build_default_cov_net(layer_input_shape=input_shape)
        if dict.encoder_type == 'FC_net':
            net = build_default_FC_net(dict.z_dim, 1024, input_shape, 'relu')
        net = tf.keras.Sequential([net, tfkl.Flatten(), tfkl.Dense(2 * dict.z_dim)])
        encoder = BasedEncoder(z_dim=dict.z_dim, net=net)
    decoder = BaseDecoder(latent_dim=dict.z_dim, num_of_labels=dict.num_of_labels)
    pre_model = []
    model_path = './mlruns/0/{}/artifacts/model/checkpoints/_1_cp'.format(dict.pre_model_path)

    if dict.label_noise_type == 'pre_defined_model':
        img_input1 = Input(shape=input_shape)
        wide_resnet_output_pre1 = wide_residual_network(img_input1, dict.num_of_labels, dict.depth,
                                                        dict.wide,
                                                        dict.weights_decay, include_top=True)
        pre_model = Model(img_input1, wide_resnet_output_pre1)
        class_loss_fn1 = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        opt1 = tf.keras.optimizers.SGD(1e-4, momentum=0.9, nesterov=True)
        pre_model.compile(optimizer=opt1, loss=class_loss_fn1, metrics='acc')
        pre_model.load_weights(model_path)
        pre_model.trainable = False
    model = VariationalNetwork(log_beta=dict.log_beta, gamma=dict.gamma, labels_dist=labels_dist,
                               encoder=encoder,
                               decoder=decoder, prior=prior,
                               loss_func_inner=loss_func_inner, beta_sched=beta_sched,
                               gamma_sched=gamma_sched,
                               labels_noise=dict.labels_noise, confusion_matrix=confusion_matrix,
                               use_logprob_func=dict.use_logprob_func,
                               file_name=log_file_path_name, label_noise_type=dict.label_noise_type,
                               pre_model=pre_model, model_path=model_path, num_of_labels=dict.num_of_labels)
    return model, scheduler


def main(argv):
    with mlflow.start_run():

        strategy = tf.distribute.MirroredStrategy()
        # with tf.device('/gpu:{}'.format(FLAGS.gpu)):
        # with strategy.scope():
        # Save params
        dict = insert_keys()

        log_file_path = mlflow.get_artifact_uri() + '/' + dict.log_file_path
        log_file_path_name = log_file_path + '/data.csv'
        print('PATH', mlflow.get_artifact_uri())
        dict = insert_keys()
        ds_train, ds_test, steps_per_epoch, steps_per_epoch_validation, confusion_matrix = get_dataset(dict)
        if dict.dataset == 'fasion_mnist':
            input_shape = (28, 28, 3)
        else:
            input_shape = (32, 32, 3)
        img_input = Input(shape=input_shape)
        if dict.run_model[0] == 'det_wide_resnet':
            wide_resnet_output = wide_residual_network(img_input, dict.num_of_labels, dict.depth, dict.wide,
                                                       dict.weights_decay, include_top=True,
                                                       is_cifar=(dict.dataset == 'cifar' or dict.dataset == 'cifar100'),
                                                       with_batchnorm=dict.with_batchnorm)
            model = Model(img_input, wide_resnet_output)
            if dict.dataset != 'cifar100':
                scheduler = schedule_cifar10_2
            else:
                scheduler = schedule_cifar100
        else:
            model, scheduler = get_vmodel(img_input, confusion_matrix, dict, log_file_path_name)

        with strategy.scope():
            metrices_list = ['accuracy']
        for name in dict.metrices:
            metrices_list.append(tf.keras.metrics.Mean(name))
        if dict.run_model[0] == 'det_wide_resnet':
            class_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        else:
            class_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        if dict.train_model:
            if dict.opt == 'adam':
                opt = tf.keras.optimizers.Adam(dict.initial_lr)
            else:
                opt = tf.keras.optimizers.SGD(dict.initial_lr, momentum=dict.momentum, nesterov=True)
            # The metrices that we want to store
            print(class_loss_fn, steps_per_epoch)
            model.compile(optimizer=opt, loss=class_loss_fn, metrics=metrices_list)
            # For debugging
            model.run_eagerly = True
            # Train
            change_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
            tb_cb = tf.keras.callbacks.TensorBoard(log_dir=log_file_path, histogram_freq=0)
            checkpoint_path = mlflow.get_artifact_uri() + "/model/checkpoints/{}_cp"
            # Save Cehckpoints of the model and csv file
            history = LoggerTrain(file_name=mlflow.get_artifact_uri() + '/c_looger.csv',
                                  checkpoint_path=checkpoint_path)
            cbks = [change_lr, tb_cb, history]
            # dict.num_of_epochs = 1
            model.fit(ds_train, steps_per_epoch=steps_per_epoch, epochs=dict.num_of_epochs, shuffle=True,
                      callbacks=cbks,
                      validation_data=ds_test,
                      validation_steps=steps_per_epoch_validation,
                      # batch_size=dict.batch_size,
                      verbose=2)
            # vals = model.predict(ds_train, steps=1)
            model.save(checkpoint_path.format(dict.num_of_epochs))
        else:
            model_path = './mlruns/0/{}/artifacts/model/checkpoints/_1_cp'.format(dict.pre_model_path)
            model.run_eagerly = True
            class_loss_fn1 = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
            opt1 = tf.keras.optimizers.SGD(1e-4, momentum=0.9, nesterov=True)
            model.compile(optimizer=opt1, loss=class_loss_fn1, metrics=metrices_list)
            model.load_weights(model_path)
            model.evaluate(ds_test)


if __name__ == '__main__':
    app.run(main)
