import numpy as np
import tensorflow as tf
import pandas as pd
from csv import  writer
def log_summary(summary_writer, epoch, matrices, logger):

    message = "-"*100 +"\n"
    message += 'Epoch {}, '.format(epoch+1)
    names, vals = [], []
    with summary_writer.as_default():
        for matric_name in matrices:
            metric_list = matrices[matric_name]
            if type(metric_list) !=list:
                metric_list = [metric_list]
            for i in range(len(metric_list)):
                metric_list_inner = metric_list[i]
                if type(metric_list_inner) != list:
                    metric_list_inner = [metric_list_inner]
                for j in range(len(metric_list_inner)):
                    metric = metric_list_inner[j]
                    tf.summary.scalar(metric.name, metric.result(), step = epoch)
                    vals.append(metric.result().numpy())
                    names.append(metric.name)
                    if metric.result()!=0:
                        message +='{} {:0.2f}, '.format(metric.name, metric.result())
                    metric.reset_states()
        logger.info(message)
    df = pd.DataFrame.from_records([vals], columns=names)
    df['epoch'] = epoch+1
    return df

def log_summary2(matrices, steps):
    names, vals = [], []
    for matric_name in matrices:
        metric_list = matrices[matric_name]
        if type(metric_list) !=list:
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
    df =pd.DataFrame(data = np.array(vals).T, columns=names)
    df['step'] = steps

    return df


def build_tensorarray_with_names(num_of_layers, num_of_clusters, num_of_epochs, name=[]):
    arr, names = [], []
    for i in  range(num_of_layers):
        inner_arr, inner_names = [], []
        for j in range(len(num_of_clusters[i])):
            arr.append((tf.TensorArray(dtype=tf.float64, size=num_of_epochs, tensor_array_name=name.format(i, j)), name.format(i, j)))
            names.append(name.format(i, j))
        #arr.append(inner_arr)
        #names.append(inner_names)
    return arr, names

def load_matrices2(num_of_layers, num_of_clusters, num_of_epochs, num_of_bins, num_of_noises):

    """Return dict of matrices for the loss and all the informatget_ixt_all_layersion measures."""
    train_loss =  (tf.TensorArray(dtype=tf.float64, size=num_of_epochs, name='train_loss'), 'train_loss')
    test_loss =  (tf.TensorArray(dtype=tf.float64, size=num_of_epochs, name='test_loss'), 'test_loss')
    train_acc =  (tf.TensorArray(dtype=tf.float64, size=num_of_epochs, name='train_acc'), 'train_acc')
    test_acc =  (tf.TensorArray(dtype=tf.float64, size=num_of_epochs, name='test_acc'), 'test_acc')

    #test_ixt_bound = [tf.TensorArray(dynamic_size=True,clear_after_read=False, dtype=tf.float64, size=num_of_epochs, tensor_array_name=r"test_i_x_y_{}".format(i)) for i in range(num_of_layers)]
    #test_ity_bound = [tf.keras.metrics.Mean(name=r"test_i_y_t_{}".format(i)) for i in range(num_of_layers)]
    test_ixt_clusterd_bound, test_ixt_clusterd_names = build_tensorarray_with_names(num_of_layers, num_of_clusters, num_of_epochs, name=r"test_i_x;t_{}_{}_c_nce")
    test_ixt_clusterd_nonlinear_bound, test_ixt_clusterd_nonlinear_names = build_tensorarray_with_names(num_of_layers, num_of_clusters, num_of_epochs, name=r"test_i_x_t_{}_{}_c_nonlinear")
    test_ity_clusterd_bound, test_ity_clusterd_names = build_tensorarray_with_names(num_of_layers, num_of_clusters, num_of_epochs, name=r"test_i_y_t_{}_{}_c_nce")
    test_ity_clusterd_mine_bound, test_ity_clusterd_mine_names = build_tensorarray_with_names(num_of_layers, num_of_clusters, num_of_epochs, name=r"test_i_y_t_{}_{}_c_mine")
    test_ixt_mine_clusterd_bound, test_ixt_mine_clusterd_names = build_tensorarray_with_names(num_of_layers, num_of_clusters, num_of_epochs, name=r"test_i_x_t_{}_{}_c_mine")
    test_ixt_dual_bound, test_ixt_dual_names = build_tensorarray_with_names(num_of_layers, num_of_clusters, num_of_epochs, name=r"test_i_x_t_{}_{}_dual")
    test_ity_dual_bound, test_ity_dual_names = build_tensorarray_with_names(num_of_layers, num_of_clusters, num_of_epochs, name=r"test_i_y_t_{}_{}_dual")

    test_ity_mine_bound, test_ity_mine_bound_name = build_tensorarray_with_names(num_of_layers,  [[1]]*num_of_layers, num_of_epochs, name=r"test_i_y_t_{}_mine")
    test_ixt_mine_bound, test_ixt_mine_bound_name = build_tensorarray_with_names(num_of_layers,  [[1]]*num_of_layers, num_of_epochs, name=r"test_i_x_t_{}_mine")

    #test_ity_mine_bound = [tf.keras.metrics.Mean(name=r"test_i_y_t_{}_mine".format(i)) for i in
    #                       range(num_of_layers)]
    #test_ixt_mine_bound = [tf.keras.metrics.Mean(name=r"test_i_x_t_{}_mine".format(i)) for i in
    #                       range(num_of_layers)]
    test_ixt_bins_bound, test_ixt_bins_bound_names = build_tensorarray_with_names(num_of_layers, [[1]*num_of_bins]*num_of_layers, num_of_epochs, name=r"testi_x_t_{}_{}_bins")
    test_ity_bins_bound, test_ity_bins_bound_names = build_tensorarray_with_names(num_of_layers,[[1]*num_of_bins]*num_of_layers, num_of_epochs, name=r"test_i_y_t_{}_{}_bins")
    test_ixt_bound, test_ixt_bound_names = build_tensorarray_with_names(num_of_layers,[[1]*num_of_noises]*num_of_layers, num_of_epochs, name=r"test_i_y_t_{}_{}_nonlinear")
    test_ity_bound, test_ity_bound_name = build_tensorarray_with_names(num_of_layers,[[1]*num_of_noises]*num_of_layers, num_of_epochs, name=r"test_i_x_t_{}_{}_nonlinear")


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
    matrices['test_ixt_nonlinear_bound'] = test_ixt_bound
    matrices['test_ixt_clusterd_bound_nce'] = test_ixt_clusterd_bound
    matrices['test_ixt_clusterd_bound_nonlinear'] = test_ixt_clusterd_nonlinear_bound
    matrices['test_ity_mine_bound'] = test_ity_mine_bound
    matrices['test_ixt_mine_bound'] = test_ixt_mine_bound
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
        if len(mat)>len(information_clustered):
            val = 0
        else:
            val =  information_clustered[i][index_1]
        te = mat[i][0]
        a = te.write(step,val)
        c = val
        df_list.append(c)
        names_list.append(mat[i][1])
        a.mark_used()
        ind+=1
    return

def insert_measure(mat, step, val, df_list, names_list):
    te = mat[0]
    a = te.write(step, val)
    a.mark_used()
    df_list.append(tf.cast(tf.constant(val), tf.float64))
    names_list.append(mat[1])

def store_data2(matrices, step, loss_value, test_loss_val, nonlinear_information, information_MINE, information_clustered,
                information_dual_ib, information_bins):
    df_list, names_list = [], []
    insert_double2(matrices['test_ixt_dual_bound'],step, information_dual_ib, 0, names_list, df_list)
    insert_double2(matrices['test_ity_dual_bound'],step, information_dual_ib, 1, names_list, df_list)
    insert_double2(matrices['test_ixt_mine_bound'], step, information_MINE, 0, names_list, df_list)
    insert_double2(matrices['test_ity_mine_bound'], step, information_MINE, 1, names_list, df_list)
    insert_double2(matrices['test_ixt_bins_bound'], step, information_bins, 0, names_list, df_list)
    insert_double2(matrices['test_ity_bins_bound'], step, information_bins, 1, names_list, df_list)
    insert_double2(matrices['test_ixt_nonlinear_bound'], step, nonlinear_information, 0, names_list, df_list)
    insert_double2(matrices['test_ity_nonlinear_bound'], step, nonlinear_information, 1, names_list, df_list)
    #[matrices['test_ixt_mine_bound'][i](information_MINE[i][0]) for i in range(len(information_MINE))]
    #[matrices['test_ity_mine_bound'][i](information_MINE[i][1]) for i in range(len(information_MINE))]

    #insert_double2(matrices['test_ixt_clusterd_bound_nce'],step, information_clustered, 0, 0, names_list, df_list)
    insert_double2(matrices['test_ixt_clusterd_bound_nonlinear'], step, information_clustered, 0, names_list, df_list)
    insert_double2(matrices['test_ity_clusterd_bound_bayes'],step,  information_clustered, 1, names_list, df_list)
    #insert_double2(matrices['test_ixt_clusterd_bound_mine'],step,  information_clustered, 0, 1, names_list, df_list)
    #insert_double2(matrices['test_ity_clusterd_bound_mine'],step,  information_clustered, 1, 1, names_list, df_list)
    #[matrices['test_ixt_bins_bound'][i](information_bins[i][1]) for i in
    # range(len(information_bins))]
    #[matrices['test_ity_bins_bound'][i](information_bins
    #                                    [i][0]) for i inrange(len(information_bins))]
    insert_measure(matrices['train_loss'], step,  loss_value[1], df_list, names_list)
    insert_measure(matrices['train_acc'], step,  loss_value[0], df_list, names_list)
    insert_measure(matrices['test_loss'], step,  test_loss_val[1], df_list, names_list)
    insert_measure(matrices['test_acc'], step,  test_loss_val[0], df_list, names_list)
    return df_list, names_list


def save_pickle(file_name, list_of_elem):
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

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
        self.train_loss.append(logs.get('loss'))
        self.train_acc.append(logs.get('accuracy'))
        if  self.step_counter in self.steps:
            val_loss, val_acc = self.model.evaluate(self.validation_data, steps =1000)
            loss, acc = self.model.evaluate(self.train_data, steps = 1000)
            list_of_elem = [self.step_counter, acc, loss,val_loss, val_acc ]
            save_pickle(self.file_name, list_of_elem)
            self.model.save(self.checkpoint_path.format(self.step_counter))
        self.step_counter+=1

def process_list(params_val):
    """Parse the given string to numbers"""
    return [int(s) for s in params_val.split('[')[1].split(']')[0].split(',')]

