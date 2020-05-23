from typing import Any
try:
    import cdsw
    use_cdsw = True
except ImportError:
    use_cdsw = False
use_cdsw = False

import numpy as np
import tensorflow as tf
import pandas as pd
from csv import writer


def log_summary(summary_writer, epoch, matrices, logger):
    message = "-" * 100 + "\n"
    message += 'Epoch {}, '.format(epoch + 1)
    names, vals = [], []
    with summary_writer.as_default():
        for matric_name in matrices:
            metric_list = matrices[matric_name]
            if type(metric_list) != list:
                metric_list = [metric_list]
            for i in range(len(metric_list)):
                metric_list_inner = metric_list[i]
                if type(metric_list_inner) != list:
                    metric_list_inner = [metric_list_inner]
                for j in range(len(metric_list_inner)):
                    metric = metric_list_inner[j]
                    tf.summary.scalar(metric.name, metric.result(), step=epoch)
                    vals.append(metric.result().numpy())
                    names.append(metric.name)
                    if metric.result() != 0:
                        message += '{} {:0.2f}, '.format(metric.name, metric.result())
                    metric.reset_states()
        logger.info(message)
    df = pd.DataFrame.from_records([vals], columns=names)
    df['epoch'] = epoch + 1
    return df


def log_summary2(matrices, steps):
    names, vals = [], []
    for matric_name in matrices:
        metric_list = matrices[matric_name]
        if type(metric_list) != list:
            metric_list = [metric_list]
        for i in range(len(metric_list)):
            metric_list_inner = metric_list[i]
            if type(metric_list_inner) != list:
                metric_list_inner = [metric_list_inner]
            for j in range(len(metric_list_inner)):
                metric = metric_list_inner[j][0]
                name = metric_list_inner[j][1]
                vals.append(metric.stack().numpy())
                names.append(name)
    df = pd.DataFrame(data=np.array(vals).T, columns=names)
    df['step'] = steps

    return df


def build_tensorarray_with_names(num_of_layers, num_of_clusters, num_of_epochs, name=[]):
    arr, names = [], []
    for i in range(num_of_layers):
        inner_arr, inner_names = [], []
        for j in range(len(num_of_clusters[i])):
            arr.append((tf.TensorArray(dtype=tf.float64, size=num_of_epochs, tensor_array_name=name.format(i, j)),
                        name.format(i, j)))
            names.append(name.format(i, j))
        # arr.append(inner_arr)
        # names.append(inner_names)
    return arr, names


def load_matrices2(num_of_layers, num_of_clusters, num_of_epochs, num_of_bins, num_of_noises):
    """Return dict of matrices for the loss and all the informatget_ixt_all_layersion measures."""
    train_loss = (tf.TensorArray(dtype=tf.float64, size=num_of_epochs, name='train_loss'), 'train_loss')
    test_loss = (tf.TensorArray(dtype=tf.float64, size=num_of_epochs, name='test_loss'), 'test_loss')
    train_acc = (tf.TensorArray(dtype=tf.float64, size=num_of_epochs, name='train_acc'), 'train_acc')
    test_acc = (tf.TensorArray(dtype=tf.float64, size=num_of_epochs, name='test_acc'), 'test_acc')

    # test_ixt_bound = [tf.TensorArray(dynamic_size=True,clear_after_read=False, dtype=tf.float64, size=num_of_epochs, tensor_array_name=r"test_i_x_y_{}".format(i)) for i in range(num_of_layers)]
    # test_ity_bound = [tf.keras.metrics.Mean(name=r"test_i_y_t_{}".format(i)) for i in range(num_of_layers)]
    test_ixt_clusterd_bound, test_ixt_clusterd_names = build_tensorarray_with_names(num_of_layers, num_of_clusters,
                                                                                    num_of_epochs,
                                                                                    name=r"test_i_x;t_{}_{}_c_nce")
    test_ixt_clusterd_nonlinear_bound, test_ixt_clusterd_nonlinear_names = build_tensorarray_with_names(num_of_layers,
                                                                                                        num_of_clusters,
                                                                                                        num_of_epochs,
                                                                                                        name=r"test_i_x_t_{}_{}_c_nonlinear")
    test_ity_clusterd_bound, test_ity_clusterd_names = build_tensorarray_with_names(num_of_layers, num_of_clusters,
                                                                                    num_of_epochs,
                                                                                    name=r"test_i_y_t_{}_{}_c_nce")
    test_ity_clusterd_bound_nonlinear, test_ity_clusterd_names_nonlinear = build_tensorarray_with_names(num_of_layers,
                                                                                                        num_of_clusters,
                                                                                                        num_of_epochs,
                                                                                                        name=r"test_i_y_t_{}_{}_c_nonlinear")
    test_ity_clusterd_mine_bound, test_ity_clusterd_mine_names = build_tensorarray_with_names(num_of_layers,
                                                                                              num_of_clusters,
                                                                                              num_of_epochs,
                                                                                              name=r"test_i_y_t_{}_{}_c_mine")
    test_ixt_mine_clusterd_bound, test_ixt_mine_clusterd_names = build_tensorarray_with_names(num_of_layers,
                                                                                              num_of_clusters,
                                                                                              num_of_epochs,
                                                                                              name=r"test_i_x_t_{}_{}_c_mine")
    test_ixt_dual_bound, test_ixt_dual_names = build_tensorarray_with_names(num_of_layers, num_of_clusters,
                                                                            num_of_epochs,
                                                                            name=r"test_i_x_t_{}_{}_dual")
    test_ity_dual_bound, test_ity_dual_names = build_tensorarray_with_names(num_of_layers, num_of_clusters,
                                                                            num_of_epochs,
                                                                            name=r"test_i_y_t_{}_{}_dual")

    test_ity_mine_bound, test_ity_mine_bound_name = build_tensorarray_with_names(num_of_layers, [[1]] * num_of_layers,
                                                                                 num_of_epochs,
                                                                                 name=r"test_i_y_t_{}_mine")
    test_ixt_mine_bound, test_ixt_mine_bound_name = build_tensorarray_with_names(num_of_layers, [[1]] * num_of_layers,
                                                                                 num_of_epochs,
                                                                                 name=r"test_i_x_t_{}_mine")

    # test_ity_mine_bound = [tf.keras.metrics.Mean(name=r"test_i_y_t_{}_mine".format(i)) for i in
    #                       range(num_of_layers)]
    # test_ixt_mine_bound = [tf.keras.metrics.Mean(name=r"test_i_x_t_{}_mine".format(i)) for i in
    #                       range(num_of_layers)]
    test_ixt_bins_bound, test_ixt_bins_bound_names = build_tensorarray_with_names(num_of_layers,
                                                                                  [[1] * num_of_bins] * num_of_layers,
                                                                                  num_of_epochs,
                                                                                  name=r"testi_x_t_{}_{}_bins")
    test_ity_bins_bound, test_ity_bins_bound_names = build_tensorarray_with_names(num_of_layers,
                                                                                  [[1] * num_of_bins] * num_of_layers,
                                                                                  num_of_epochs,
                                                                                  name=r"test_i_y_t_{}_{}_bins")
    test_ixt_bound, test_ixt_bound_names = build_tensorarray_with_names(num_of_layers,
                                                                        [[1] * num_of_noises] * num_of_layers,
                                                                        num_of_epochs,
                                                                        name=r"test_i_x_t_{}_{}_nonlinear")
    test_ity_bound, test_ity_bound_name = build_tensorarray_with_names(num_of_layers,
                                                                       [[1] * num_of_noises] * num_of_layers,
                                                                       num_of_epochs,
                                                                       name=r"test_i_y_t_{}_{}_nonlinear")

    test_ixt_bound_normalized, test_ixt_bound_names_normalized = build_tensorarray_with_names(num_of_layers,
                                                                                              [[
                                                                                                   1] * num_of_noises] * num_of_layers,
                                                                                              num_of_epochs,
                                                                                              name=r"test_i_x_t_{}_{}_nonlinear_norm")
    test_ity_bound_normlized, test_ity_bound_name_normalized = build_tensorarray_with_names(num_of_layers,
                                                                                            [[
                                                                                                 1] * num_of_noises] * num_of_layers,
                                                                                            num_of_epochs,
                                                                                            name=r"test_i_y_t_{}_{}_nonlinear_norm")

    matrices = {}
    matrices['train_loss'] = train_loss
    matrices['train_acc'] = train_acc
    matrices['test_loss'] = test_loss
    matrices['test_acc'] = test_acc
    matrices['test_ixt_bins_bound'] = test_ixt_bins_bound
    matrices['test_ity_bins_bound'] = test_ity_bins_bound
    matrices['test_ixt_dual_bound'] = test_ixt_dual_bound
    matrices['test_ity_dual_bound'] = test_ity_dual_bound
    matrices['test_ity_nonlinear_bound'] = test_ity_bound
    matrices['test_ity_clusterd_bound_bayes'] = test_ity_clusterd_bound
    matrices['test_ity_clusterd_bound_nonlinear'] = test_ity_clusterd_bound_nonlinear
    matrices['test_ixt_nonlinear_bound'] = test_ixt_bound
    matrices['test_ixt_clusterd_bound_nce'] = test_ixt_clusterd_bound
    matrices['test_ixt_clusterd_bound_nonlinear'] = test_ixt_clusterd_nonlinear_bound
    matrices['test_ity_mine_bound'] = test_ity_mine_bound
    matrices['test_ixt_mine_bound'] = test_ixt_mine_bound
    matrices['test_ity_normelized_kde_bound'] = test_ity_bound_normlized
    matrices['test_ixt_normelized_kde_bound'] = test_ixt_bound_normalized
    matrices['test_ity_clusterd_bound_mine'] = test_ity_clusterd_mine_bound
    matrices['test_ixt_clusterd_bound_mine'] = test_ixt_mine_clusterd_bound
    return matrices


def insert_double(mat, information_clustered, index_1, index_2):
    [[mat[i][j][0](information_clustered[i][j][index_1][index_2]) for j in
      range(len(information_clustered[i]))] for i in range(len(information_clustered))]


def insert_double2(mat, step, information_clustered, index_1, names_list, df_list):
    shape_c = len(mat)
    ind = 0
    for i in range(shape_c):
        if len(mat) > len(information_clustered):
            val = 0
        else:
            val = information_clustered[i][index_1]
        te = mat[i][0]
        a = te.write(step, val)
        c = val
        df_list.append(c)
        names_list.append(mat[i][1])
        a.mark_used()
        ind += 1
    return


def insert_measure(mat, step, val, df_list, names_list):
    te = mat[0]
    a = te.write(step, val)
    a.mark_used()
    df_list.append(tf.cast(tf.constant(val), tf.float64))
    names_list.append(mat[1])


def store_data2(matrices, step, loss_value, test_loss_val, nonlinear_information, information_MINE,
                information_clustered,
                normalized_information_KDE,
                information_dual_ib, information_bins):
    df_list, names_list = [], []
    insert_double2(matrices['test_ixt_dual_bound'], step, information_dual_ib, 0, names_list, df_list)
    insert_double2(matrices['test_ity_dual_bound'], step, information_dual_ib, 1, names_list, df_list)
    insert_double2(matrices['test_ixt_mine_bound'], step, information_MINE, 0, names_list, df_list)
    insert_double2(matrices['test_ity_mine_bound'], step, information_MINE, 1, names_list, df_list)
    insert_double2(matrices['test_ixt_normelized_kde_bound'], step, normalized_information_KDE, 0, names_list, df_list)
    insert_double2(matrices['test_ity_normelized_kde_bound'], step, normalized_information_KDE, 1, names_list, df_list)
    insert_double2(matrices['test_ixt_bins_bound'], step, information_bins, 0, names_list, df_list)
    insert_double2(matrices['test_ity_bins_bound'], step, information_bins, 1, names_list, df_list)
    insert_double2(matrices['test_ixt_nonlinear_bound'], step, nonlinear_information, 0, names_list, df_list)
    insert_double2(matrices['test_ity_nonlinear_bound'], step, nonlinear_information, 1, names_list, df_list)
    # [matrices['test_ixt_mine_bound'][i](information_MINE[i][0]) for i in range(len(information_MINE))]
    # [matrices['test_ity_mine_bound'][i](information_MINE[i][1]) for i in range(len(information_MINE))]

    # insert_double2(matrices['test_ixt_clusterd_bound_nce'],step, information_clustered, 0, 0, names_list, df_list)
    insert_double2(matrices['test_ixt_clusterd_bound_nonlinear'], step, information_clustered, 0, names_list, df_list)
    insert_double2(matrices['test_ity_clusterd_bound_bayes'], step, information_clustered, 1, names_list, df_list)
    insert_double2(matrices['test_ity_clusterd_bound_nonlinear'], step, information_clustered, 2, names_list, df_list)
    # insert_double2(matrices['test_ixt_clusterd_bound_mine'],step,  information_clustered, 0, 1, names_list, df_list)
    # insert_double2(matrices['test_ity_clusterd_bound_mine'],step,  information_clustered, 1, 1, names_list, df_list)
    # [matrices['test_ixt_bins_bound'][i](information_bins[i][1]) for i in
    # range(len(information_bins))]
    # [matrices['test_ity_bins_bound'][i](information_bins
    #                                    [i][0]) for i inrange(len(information_bins))]
    insert_measure(matrices['train_loss'], step, loss_value[1], df_list, names_list)
    insert_measure(matrices['train_acc'], step, loss_value[0], df_list, names_list)
    insert_measure(matrices['test_loss'], step, test_loss_val[1], df_list, names_list)
    insert_measure(matrices['test_acc'], step, test_loss_val[0], df_list, names_list)
    return df_list, names_list


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
        self.step_counter = 0
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


class LoggerHistory(tf.keras.callbacks.Callback):
    """Save loss and accuracy to csv and model checkpoints"""

    def __init__(self, val_data, train_data, steps, file_name, checkpoint_path):
        super().__init__()
        self.validation_data = val_data
        self.steps = steps
        self.train_data = train_data
        self.step_counter = 0
        self.file_name = file_name
        self.checkpoint_path = checkpoint_path
        list_of_elem = ['step', 'acc', 'loss', 'val_loss', 'val_acc']
        save_pickle(self.file_name, list_of_elem)

    def on_epoch_begin(self, epoch, logs=None):
        self.train_loss = []
        self.train_acc = []

    def on_batch_end(self, batch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs=None):
        self.train_loss.append(logs.get('loss'))
        self.train_acc.append(logs.get('accuracy'))
        if self.step_counter in self.steps:
            # val_loss, val_acc = self.model.evaluate(self.validation_data, steps =1000)
            # loss, acc = self.model.evaluate(self.train_data, steps = 1000)
            # list_of_elem = [self.step_counter]
            # save_pickle(self.file_name, list_of_elem)
            self.model.save(self.checkpoint_path.format(self.step_counter))
        self.step_counter += 1


def process_list(params_val):
    """Parse the given string to numbers"""
    return [int(s) for s in params_val.split('[')[1].split(']')[0].split(',')]


def tf_unique_2d(self, x):
    x_shape = tf.shape(x)  # (3,2)
    x1 = tf.tile(x, [1, x_shape[0]])  # [[1,2],[1,2],[1,2],[3,4],[3,4],[3,4]..]
    x2 = tf.tile(x, [x_shape[0], 1])  # [[1,2],[1,2],[1,2],[3,4],[3,4],[3,4]..]

    x1_2 = tf.reshape(x1, [x_shape[0] * x_shape[0], x_shape[1]])
    x2_2 = tf.reshape(x2, [x_shape[0] * x_shape[0], x_shape[1]])
    cond = tf.reduce_all(tf.equal(x1_2, x2_2), axis=1)
    cond = tf.reshape(cond, [x_shape[0], x_shape[0]])  # reshaping cond to match x1_2 & x2_2
    cond_shape = tf.shape(cond)
    cond_cast = tf.cast(cond, tf.int32)  # convertin condition boolean to int
    cond_zeros = tf.zeros(cond_shape, tf.int32)  # replicating condition tensor into all 0's

    # CREATING RANGE TENSOR
    r = tf.range(x_shape[0])
    r = tf.add(tf.tile(r, [x_shape[0]]), 1)
    r = tf.reshape(r, [x_shape[0], x_shape[0]])

    # converting TRUE=1 FALSE=MAX(index)+1 (which is invalid by default) so when we take min it wont get selected & in end we will only take values <max(indx).
    f1 = tf.multiply(tf.ones(cond_shape, tf.int32), x_shape[0] + 1)
    f2 = tf.ones(cond_shape, tf.int32)
    cond_cast2 = tf.where(tf.equal(cond_cast, cond_zeros), f1, f2)  # if false make it max_index+1 else keep it 1

    # multiply range with new int boolean mask
    r_cond_mul = tf.multiply(r, cond_cast2)
    r_cond_mul2 = tf.reduce_min(r_cond_mul, axis=1)
    r_cond_mul3, unique_idx = tf.unique(r_cond_mul2)
    r_cond_mul4 = tf.subtract(r_cond_mul3, 1)

    # get actual values from unique indexes
    op = tf.gather(x, r_cond_mul4)

    return (op)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d
