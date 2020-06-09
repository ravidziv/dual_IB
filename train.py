"""Train  a  family of variational models - ceb, vib and the dual ib """
use_cdsw = False
import os
import sys

sys.path.insert(0, "/home/cdsw/")
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import mlflow
from VdualIB.utils import LoggerTrain, insert_keys
from absl import flags
from absl import app
import tensorflow_probability as tfp
from functools import partial
from VdualIB.schedulers import *
from VdualIB.models.models import BasePrior, BaseDecoder, BasedEncoder, BZYPrior, BaseLabelsModel, build_default_FC_net, \
    build_default_cov_net
from VdualIB.models.wide_resnet import wide_residual_network
from VdualIB.variational_network import VariationalNetwork
from VdualIB.datasets import load_cifar_data, load_mnist_data, load_fasion_mnist_data, load_cifar100_data
import mlflow.tensorflow

tfkl = tf.keras.layers
dfd = tfp.distributions
FLAGS = flags.FLAGS

flags.DEFINE_integer('end_anneling', 4000, 'In which step to stop annealing beta')

flags.DEFINE_integer('z_dim', 320, 'The dimension of the hidden space')
flags.DEFINE_integer('h_dim', 10, 'the dimension of the hidden layers of the encoder')
flags.DEFINE_integer('batch_size', 64, 'For training')
flags.DEFINE_integer('num_of_epochs', 10000, 'For training')
flags.DEFINE_string('activation', 'relu', 'Activation of the encoder layers')
flags.DEFINE_string('log_file_path', 'w_resnet', 'Where to save the network tensorboard files')
flags.DEFINE_string('dataset', 'cifar', 'Which dataset to load (mnist, fasion_mnist, cifar, cifar100)')
flags.DEFINE_string('encoder_type', 'wide_resnet', 'The encoder of the network - [wide_resnet, efficientNetB0, '
                                                   'covnet, FC]')
flags.DEFINE_string('opt', 'adam', 'Which learning method to train with')

flags.DEFINE_multi_string('run_model', 'var',
                          'Which model to run - var/det - variational or deterministic')
flags.DEFINE_float('initial_lr', 1e-4, 'The lr for the train')
flags.DEFINE_float('momentum', 0.9, 'The momentum of the SGD')
flags.DEFINE_float('beta', 0., 'beta - the coefficient for hyz - for fure dualIB it is 0')
flags.DEFINE_float('gamma', 1000., 'the coefficient for hyhatz - for fure VIB/CEB it is 0')
flags.DEFINE_float('weights_decay', 0.0005, 'The weights decay training')
flags.DEFINE_float('c', 1e-2, 'The noise for calculating the kl of the labels in the dual ib')
flags.DEFINE_string('label_noise_type', 'confusion_matrix_noise',
                    ['confusion_matrix_noise, gaussian_noise', 'smooth_noise', 'pre_defined_model'])
flags.DEFINE_integer('depth', 28, 'the depth of the wide resent newtork')
flags.DEFINE_integer('wide', 10, 'the wide of the wide resent newtork')
flags.DEFINE_integer('num_of_train', -1, 'The number of train examples for the training, -1 for all')
flags.DEFINE_integer('start_val_anneling', 100, 'The initial value to anneling beta/gamma')
flags.DEFINE_integer('gpu', 0, 'gpu index')
flags.DEFINE_integer('train_verbose', 2, 'The verbose mode of the train')
flags.DEFINE_multi_string('metrices',
                          ['hzy', 'hzx', 'hyz', 'hyhatz', 'ixt', 'izy', 'izyhat', 'hyz_noisy', 'total_loss', 'beta',
                           'gamma'],
                          'The metrices to save')
flags.DEFINE_bool("noisy_learning", True, 'Train ceb/vib models with noisy labels')
flags.DEFINE_bool("use_logprob_func", False, 'Sample label noise from gaussian or empirical mle')
flags.DEFINE_string('pre_model_path', 'e226f66bbee1423e911ea63841c6d80e',
                    'id run for pre-train model for the noise model')
flags.DEFINE_string('prior', 'ceb_prior', 'base/ceb prior to use for in the network')


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


def get_encoder(input_shape, dict):
    if dict.encoder_type == 'wide_resnet':
        net = wide_residual_network(input_shape, dict.num_of_labels, dict.depth, dict.wide,
                                    dict.weights_decay,
                                    is_cifar=(dict.dataset == 'cifar' or dict.dataset == 'cifar100'))
    if dict.encoder_type == 'efficientNetB0':
        net = tf.keras.applications.EfficientNetB0(
            include_top=False, weights=None, input_shape=input_shape,
            pooling=None, classes=dict.num_of_labels)
    if dict.encoder_type == 'covnet':
        net = build_default_cov_net(layer_input_shape=input_shape)
    if dict.encoder_type == 'FC_net':
        net = build_default_FC_net(dict.z_dim, dict.h_dim, input_shape, dict.activation)
    net = tf.keras.Sequential([net, tfkl.Flatten(), tfkl.Dense(2 * dict.z_dim)])
    return net


def get_vmodel(input_shape, confusion_matrix, dict, log_file_path_name):
    # Create encoder, decoder and prior for variational network
    model_path = './mlruns/0/{}/artifacts/model/checkpoints/_1_cp'.format(dict.pre_model_path)
    if dict.prior == 'base':
        prior = BasePrior(z_dims=dict.z_dim)
    else:
        prior = BZYPrior(z_dims=dict.z_dim, num_classes=dict.num_of_labels)
    labels_dist = BaseLabelsModel(confusion_matrix=confusion_matrix, num_claases=dict.num_of_labels)
    beta_sched = partial(lerp, start_step=0, end_step=dict.end_anneling, start_val=dict.start_val_anneling,
                         end_val=dict.beta)
    gamma_sched = partial(lerp, start_step=0, end_step=dict.end_anneling, start_val=dict.start_val_anneling,
                          end_val=dict.gamma)
    if dict.dataset == 'cifar' or dict.dataset == 'cifar100':
        lr_scheduler = scheduler_cifar
    else:
        lr_scheduler = scheduler_mnist

    net = get_encoder(input_shape=input_shape, dict=dict)
    encoder = BasedEncoder(z_dim=dict.z_dim, net=net)
    decoder = BaseDecoder(latent_dim=dict.z_dim, num_of_labels=dict.num_of_labels)
    pre_model = []
    # If we neet to build and compile a trained model for the labels
    if dict.label_noise_type == 'pre_defined_model':
        pre_model = wide_residual_network(input_shape, dict.num_of_labels, dict.depth,
                                          dict.wide,
                                          dict.weights_decay, include_top=True)
        class_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        pre_model.compile(optimizer='sgd', loss=class_loss_fn, metrics='acc')
        pre_model.load_weights(model_path)
        pre_model.trainable = False
    model = VariationalNetwork(beta=dict.beta, gamma=dict.gamma, labels_dist=labels_dist,
                               encoder=encoder,
                               decoder=decoder, prior=prior, beta_sched=beta_sched,
                               gamma_sched=gamma_sched,
                               labels_noise=dict.labels_noise, confusion_matrix=confusion_matrix,
                               use_logprob_func=dict.use_logprob_func,
                               file_name=log_file_path_name, label_noise_type=dict.label_noise_type,
                               pre_model=pre_model, model_path=model_path, num_of_labels=dict.num_of_labels)
    return model, lr_scheduler


def get_det_model(input_shape, dict):
    """Return deterministic wide resent network"""
    model = wide_residual_network(input_shape, dict.num_of_labels, dict.depth, dict.wide,
                                  dict.weights_decay, include_top=True,
                                  is_cifar=(dict.dataset == 'cifar' or dict.dataset == 'cifar100'))
    if dict.dataset != 'cifar100':
        scheduler = schedule_cifar10_2
    else:
        scheduler = schedule_cifar100

    return model, scheduler


def get_paths(dict):
    log_file_path = os.path.join(mlflow.get_artifact_uri(), dict.log_file_path)
    log_file_path_name = os.path.join(log_file_path, 'data.csv')
    checkpoint_path = os.path.join(mlflow.get_artifact_uri(), "model/checkpoints/{}_cp")
    logger_path = os.path.join(mlflow.get_artifact_uri(), 'c_looger.csv')
    return log_file_path, log_file_path_name, checkpoint_path, logger_path


def main(argv):
    with mlflow.start_run():
        # Save params
        dict = insert_keys(FLAGS)
        ds_train, ds_test, steps_per_epoch, steps_per_epoch_validation, confusion_matrix = get_dataset(dict)
        log_file_path, log_file_path_name, checkpoint_path, logger_path = get_paths(dict)
        if dict.dataset == 'cifar100':
            dict.num_of_labels = 100
        else:
            dict.num_of_labels = 10
        if dict.dataset == 'fasion_mnist':
            input_shape = (28, 28, 3)
        else:
            input_shape = (32, 32, 3)
        if dict.run_model == 'det':
            model, scheduler = get_det_model(input_shape, dict)
        else:
            model, scheduler = get_vmodel(input_shape, confusion_matrix, dict, log_file_path_name)
        # The metrices that we want to store
        metrices_list = ['accuracy']
        for name in dict.metrices:
            metrices_list.append(tf.keras.metrics.Mean(name))
        class_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        # Optimizer
        if dict.opt == 'adam':
            opt = tf.keras.optimizers.Adam(dict.initial_lr)
        else:
            opt = tf.keras.optimizers.SGD(dict.initial_lr, momentum=dict.momentum, nesterov=True)
        model.compile(optimizer=opt, loss=class_loss_fn, metrics=metrices_list)
        # For debugging
        model.run_eagerly = True
        change_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
        tb_cb = tf.keras.callbacks.TensorBoard(log_dir=log_file_path, histogram_freq=0)
        # Save checkpoints of the model and csv file
        history = LoggerTrain(file_name=logger_path, checkpoint_path=checkpoint_path)
        cbks = [change_lr, tb_cb, history]
        # Train
        model.fit(ds_train,
                  steps_per_epoch=5,
                  epochs=dict.num_of_epochs, shuffle=True,
                  callbacks=cbks,
                  validation_data=ds_test,
                  validation_steps=steps_per_epoch_validation,
                  batch_size=dict.batch_size,
                  verbose=FLAGS.train_verbose)
        model.save(checkpoint_path.format(dict.num_of_epochs))


if __name__ == '__main__':
    app.run(main)
