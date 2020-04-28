"""Implement a  family of variational models - ceb, vib and the dual ib """
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import mlflow
import tensorflow as tf
from absl import flags
from absl import app
import tensorflow_probability as tfp

from VdualIB.models.models import BasePrior, BaseDecoder, BasedEncoder, BZYPrior
from VdualIB.models.wide_resnet import wide_residual_network
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from VdualIB.variational_network import loss_func_ib, loss_func_dual_ib, VariationalNetwork
from datasets import  load_cifar_data, load_mnist_data
tfkl = tf.keras.layers
dfd = tfp.distributions
FLAGS = flags.FLAGS
import mlflow.tensorflow

flags.DEFINE_integer('z_dim', 320, 'The dimension of the hidden space')
flags.DEFINE_integer('num_of_labels', 10, 'The number of labels')
flags.DEFINE_integer('h_dim', 10, 'the dimension of the hidden layers of the encoder')
flags.DEFINE_integer('batch_size', 128, 'For training')
flags.DEFINE_integer('num_of_epochs', 1500, 'For training')
flags.DEFINE_string('activation', 'relu', 'Activation of the encoder layers')
flags.DEFINE_string('log_file_path', './w_resnet/', 'Where to save the network tensorboard')
flags.DEFINE_string('dataset', 'cifar', 'Which dataset to load')

flags.DEFINE_multi_enum('run_model', 'dual_ib', ['vib', 'ceb','dual_ib'],  'Which model to run')
flags.DEFINE_float('initial_lr', 1e-4, 'The lr for the train')
flags.DEFINE_float('log_beta',200., 'log_beta value for the loss function')
flags.DEFINE_float('weights_decay',0.0005, 'The weights decay training')
flags.DEFINE_float('labels_noise',1e04, 'The noise for calculating the kl of the labels in the dual ib')
flags.DEFINE_integer('depth',28, 'the depth of the wide resent newtork')
flags.DEFINE_integer('wide',10, 'the wide of the wide resent newtork')
flags.DEFINE_multi_string('metrices', ['hzy', 'hzx', 'hyz', 'hyhatz', 'ixt', 'izy', 'izyhat', 'total_loss'], 'The metrices to save')


def scheduler_mnist(epoch, lr):
    return lr


def scheduler_cifar(epoch, lr):
    if epoch < 500:
        return 1e-3
    if epoch ==100:
        return lr*0.3
    if epoch ==1250:
        return lr*0.3
    return 0.0008


def beta_sched_cifar(log_beta, step):
    if step<300:
        return tf.maximum(100., log_beta)
    elif step<600:
        return tf.maximum(2., log_beta)
    else:
        return log_beta

def beta_sched_mnist(log_beta, step):
    if step<300:
        return 100
    elif step<600:
        return 50
    else:
        return log_beta


#mlflow.tensorflow.autolog(1)

def main(argv):
    # Save params
    with mlflow.start_run():
        #Save params to logger
        for key in FLAGS.flag_values_dict():
            try:
                value = FLAGS.get_flag_value(key, None)
                mlflow.log_param(key, value)
            except:
                continue
        if FLAGS.dataset == 'mnist':
            ds_train, ds_test, steps_per_epoch = load_mnist_data(batch_size=FLAGS.batch_size, num_epochs = FLAGS.num_of_epochs)
        else:
            ds_train, ds_test, steps_per_epoch = load_cifar_data(num_class=FLAGS.num_of_labels, batch_size=FLAGS.batch_size)
        # Create encoder, decoder and the model
        if FLAGS.run_model[0] =='ceb' or FLAGS.run_model[0] == 'vib':
            loss_func_inner = loss_func_ib
            if FLAGS.run_model =='ceb':
                prior = BZYPrior(z_dims=FLAGS.z_di)
            else:
                prior = BasePrior(z_dims=FLAGS.z_dim)
        elif FLAGS.run_model[0] == 'dual_ib':
            loss_func_inner = loss_func_dual_ib
            prior = BasePrior(z_dims=FLAGS.z_dim)
        #change it to use multigpu training
        strategy = tf.distribute.MirroredStrategy()
        if True:
            prior.build([FLAGS.batch_size,])
            if FLAGS.dataset =='mnist':
                encoder = BasedEncoder(z_dim=FLAGS.z_dim, h_dim=FLAGS.h_dim, activation=FLAGS.activation,
                                       layer_input_shape=(28, 28, 1))
                beta_sched = beta_sched_mnist
                scheduler = scheduler_mnist

            else:
                img_input = Input(shape=ds_train._flat_shapes[0][1:])
                output = wide_residual_network(img_input, FLAGS.num_of_labels, FLAGS.depth, FLAGS.wide,
                                               FLAGS.weights_decay)
                beta_sched = beta_sched_cifar
                scheduler = scheduler_cifar
                net = Model(img_input, output)
                encoder = BasedEncoder(z_dim=FLAGS.z_dim, net=net )
            decoder = BaseDecoder(latent_dim = FLAGS.z_dim, num_of_labels=FLAGS.num_of_labels)
            model = VariationalNetwork(log_beta=FLAGS.log_beta, encoder=encoder, decoder=decoder, prior=prior,
                                       loss_func_inner=loss_func_inner, beta_sched=beta_sched,
                                       labels_noise=FLAGS.labels_noise)

            class_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
            opt = tf.keras.optimizers.Adam(FLAGS.initial_lr)
            #The metrices that we want to store
            metrices_list = ['accuracy']
            for name in FLAGS.metrices:
                metrices_list.append(tf.keras.metrics.Mean(name))
            model.compile(optimizer=opt, loss=class_loss_fn, metrics=metrices_list)
            #For debugging
            model.run_eagerly = True
            #Train
            log_file_path = mlflow.get_artifact_uri() + './'+ FLAGS.log_file_path
            change_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
            tb_cb = tf.keras.callbacks.TensorBoard(log_dir=log_file_path, histogram_freq=0)
            cbks = [change_lr, tb_cb]
            model.build(input_shape=ds_train._flat_shapes[0][:])
            model.fit(ds_train, steps_per_epoch=steps_per_epoch, epochs=FLAGS.num_of_epochs,
                   callbacks=[cbks],
                   validation_data=ds_test,
                  batch_size=FLAGS.batch_size,
                      verbose=1)

if __name__ == '__main__':
    app.run(main)
