"""Scheduleras file for different training"""
import tensorflow as tf


@tf.function
def lerp(global_step, start_step, end_step, start_val, end_val):
    """Utility function to linearly interpolate two values."""
    interp = (tf.cast(global_step - start_step, tf.float32)
              / tf.cast(end_step - start_step, tf.float32))
    interp = tf.maximum(0.0, tf.minimum(1.0, interp))
    return start_val * (1.0 - interp) + end_val * interp


def scheduler_mnist(epoch, lr):
    return lr


lr_schedule = [60, 120, 160]  # epoch_step


def schedule_cifar10_2(epoch_idx):
    if (epoch_idx + 1) < lr_schedule[0]:
        return 0.1
    elif (epoch_idx + 1) < lr_schedule[1]:
        return 0.02  # lr_decay_ratio = 0.2
    elif (epoch_idx + 1) < lr_schedule[2]:
        return 0.004
    return 0.0008


def scheduler_cifar(epoch, lr):
    if epoch == 300:
        lr = lr * 0.5
    if epoch == 400:
        lr = lr * 0.5
    if epoch == 500:
        lr = lr * 0.5
    tf.summary.scalar('learning rate', data=lr, step=epoch)
    return lr


def scheduler_cifar_n(epoch, lr):
    tf.summary.scalar('learning rate', data=lr, step=epoch)
    return lr
    if epoch == 0:
        return lr
    if epoch > 0 and epoch < 60:
        return 1e-7
    elif epoch >= 60 and epoch < 120:
        return 0.000002
    elif epoch >= 120 and epoch < 160:
        return 0.000004
    return 0.0000008


@tf.function
def betloss_func_iba_sched_cifar(log_beta, step):
    if step < 4000:
        n_log_beta = tf.maximum(100., log_beta)
    elif step > 500 and step > 800:
        n_log_beta = tf.maximum(2., log_beta)
    else:
        n_log_beta = log_beta
    return n_log_beta


def beta_sched_mnist(log_beta, step):
    if step < 300:
        return 100
    elif step < 600:
        return 2
    else:
        return log_beta
