import numpy as np
import tensorflow as tf

def log_summary(summary_writer, optimizer, epoch, matrices, logger):
    message = 'Epoch {}, '.format(epoch+1)
    with summary_writer.as_default():
        for matric_name in matrices:
            metric_list = matrices[matric_name]
            if type(metric_list) !=list:
                metric_list = [metric_list]
            for i in range(len(metric_list)):
                metric = metric_list[i]
                tf.summary.scalar(metric.name, metric.result(), step = optimizer.iterations)
                message +='{} {:0.3f}, '.format(metric.name, metric.result())
                metric.reset_states()
        logger.info(message)

def load_matrices(model):
    """Return dict of matrices for the loss and all the information measures."""
    train_loss = tf.keras.metrics.Mean(name='Train loss')
    test_loss = tf.keras.metrics.Mean(name='Test loss')
    test_ixt_bound = [tf.keras.metrics.Mean(name=r"Test I(X;T_{})".format(i)) for i in range(len(model.layers))]
    test_ity_bound = [tf.keras.metrics.Mean(name=r"Test I(Y;T_{})".format(i)) for i in range(len(model.layers))]
    test_ixt_clusterd_bound = [tf.keras.metrics.Mean(name=r"Test I(X;T_{})_c_nce".format(i)) for i in
                               range(len(model.layers))]
    test_ity_clusterd_bound = [tf.keras.metrics.Mean(name=r"Test I(Y;T_{})_c_nce".format(i)) for i in
                               range(len(model.layers))]
    test_ixt_dual_bound = [tf.keras.metrics.Mean(name=r"Test I(X;T_{})_dual".format(i)) for i in
                           range(len(model.layers))]
    test_ity_dual_bound = [tf.keras.metrics.Mean(name=r"Test I(Y;T_{})_dual".format(i)) for i in
                           range(len(model.layers))]
    test_ity_mine_bound = [tf.keras.metrics.Mean(name=r"Test I(Y;T_{})_mine".format(i)) for i in
                           range(len(model.layers))]
    test_ixt_mine_bound = [tf.keras.metrics.Mean(name=r"Test I(X;T_{})_mine".format(i)) for i in
                           range(len(model.layers))]
    test_ity_clusterd_mine_bound = [tf.keras.metrics.Mean(name=r"Test I(Y;T_{})_c_mine".format(i)) for i in
                           range(len(model.layers))]
    test_ixt_mine_clusterd_bound = [tf.keras.metrics.Mean(name=r"Test I(X;T_{})_c_mine".format(i)) for i in
                           range(len(model.layers))]
    test_ixt_bins_bound = [tf.keras.metrics.Mean(name=r"Test I(X;T_{})_bins".format(i)) for i in
                                    range(len(model.layers))]
    test_ity_bins_bound = [tf.keras.metrics.Mean(name=r"Test I(Y;T_{})_bins".format(i)) for i in
                                    range(len(model.layers))]
    matrices = {}
    matrices['train_loss'] = train_loss
    matrices['test_loss'] = test_loss
    matrices['test_ixt_bins_bound'] = test_ixt_bins_bound
    matrices['test_ity_bins_bound'] = test_ity_bins_bound
    matrices['test_ixt_dual_bound'] = test_ixt_dual_bound
    matrices['test_ity_dual_bound'] = test_ity_dual_bound
    matrices['test_ity_linear_bound'] = test_ity_bound
    matrices['test_ity_clusterd_bound_nce'] = test_ity_clusterd_bound
    matrices['test_ixt_linear_bound'] = test_ixt_bound
    matrices['test_ixt_clusterd_bound_nce'] = test_ixt_clusterd_bound
    matrices['test_ity_mine_bound'] = test_ity_mine_bound
    matrices['test_ixt_mine_bound'] = test_ixt_mine_bound
    matrices['test_ity_clusterd_bound_mine'] = test_ity_clusterd_mine_bound
    matrices['test_ixt_clusterd_bound_mine'] = test_ixt_mine_clusterd_bound
    return matrices

def store_data(matrices, loss_value, test_loss_val, linear_information, information_MINE, information_clustered,
               information_dual_ib, information_bins):
    [matrices['test_ixt_dual_bound'][i](information_dual_ib[i][0]) for i in range(len(information_dual_ib))]
    [matrices['test_ity_dual_bound'][i](information_dual_ib[i][1]) for i in range(len(information_dual_ib))]

    [matrices['test_ixt_mine_bound'][i](information_MINE[i][0]) for i in range(len(information_MINE))]
    [matrices['test_ity_mine_bound'][i](information_MINE[i][1]) for i in range(len(information_MINE))]

    [matrices['test_ixt_linear_bound'][i](linear_information[0][i]) for i in range(len(linear_information[0]))]
    [matrices['test_ity_linear_bound'][i](linear_information[1][i]) for i in range(len(linear_information[1]))]

    [matrices['test_ixt_clusterd_bound_nce'][i](information_clustered[0][i][0]) for i in range(len(information_clustered[0]))]
    [matrices['test_ity_clusterd_bound_nce'][i](information_clustered[1][i][0]) for i in range(len(information_clustered[1]))]
    [matrices['test_ixt_clusterd_bound_mine'][i](information_clustered[0][i][1]) for i in range(len(information_clustered[0]))]
    [matrices['test_ity_clusterd_bound_mine'][i](information_clustered[1][i][1]) for i in range(len(information_clustered[1]))]

    [matrices['test_ixt_bins_bound'][i](information_bins[i][1]) for i in
     range(len(information_bins))]
    [matrices['test_ity_bins_bound'][i](information_bins
                                        [i][0]) for i in
     range(len(information_bins))]

    matrices['train_loss'](loss_value)
    matrices['test_loss'](test_loss_val)