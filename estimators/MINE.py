"""MINE (Mutual Information Neural Estimator), from the paper: https://arxiv.org/abs/1801.04062."""
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
tfkl = tf.keras.layers
from sklearn.model_selection import train_test_split

def mlp(hidden_dim, output_dim, layers, activation):
    layers_list = []
    for layer_inde in range(layers):
        layers_list.append(tfkl.Dense(hidden_dim, activation))
        layers_list.append(tfkl.Dropout(0.5))
    layers_list.append(tfkl.Dense(output_dim))
    return tf.keras.Sequential(layers_list)


def mine_loss(t_xy, t_xy_bar):
    log_term = tf.math.log(tf.cast(t_xy.shape[0], tf.float32))
    merg_term =tf.exp(tf.reduce_logsumexp(t_xy_bar)-log_term)
    joint_term =tf.reduce_mean(t_xy)
    loss = (1. +joint_term- merg_term)
    return -loss


def train_mine(xs, ts, batch_size=50, epochs=25, lr = 1e-3):
    """return the information between x and t by the MINE estimator."""
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        es_callbatck = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0, patience=1, verbose=2, mode='auto',
            baseline=3, restore_best_weights=False
        )
        mine_model = MINE()
        #The zero loss is a workaround to remove the warning
        sgd = tf.keras.optimizers.Adam(lr)
        mine_model.compile(optimizer=sgd, loss = {'output_1': lambda x,y : 0.0})
        mine_model.run_eagerly = True
        X_train, X_test, y_train, y_test = train_test_split( xs.numpy(),ts.numpy(), test_size = 0.3)
        dataset_train = tf.data.Dataset.from_tensor_slices(({"x": X_train, "y": y_train}, y_train)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).repeat()
        dataset_test = tf.data.Dataset.from_tensor_slices(({"x": X_test, "y": y_test}, y_test)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).repeat()
        fit_history = mine_model.fit(dataset_train, validation_data = dataset_test,  steps_per_epoch =1000, validation_steps=50 , batch_size=batch_size, epochs=epochs,
                                     verbose=2, callbaks = [es_callbatck])
        fit_loss = np.min(np.array(fit_history.history['val_loss']))
        #EMA_SPAN = 5
        #mis = np.array(fit_history.history['val_loss']) * -1
        #mis_smooth = pd.Series(mis).ewm(span=EMA_SPAN).mean()
        #plt.figure()
        #p1 = plt.plot(mis, alpha=0.3)[0]
        #plt.plot(mis_smooth, c=p1.get_color())
        #plt.show()
    return -tf.cast(fit_loss, tf.float64)



def information_mine(px, xs, ts_layers, py_x, ixy, batch_size, epochs, num_of_samples=10):
    # Get I(X;T) I(Y;T) for each layer with the MINE estimator
    information = []
    ixt = 0
    hx = px.entropy()
    information.append([tf.cast(hx, tf.float64),tf.cast(ixy, tf.float64)])
    ys = tf.cast(tf.reshape(tf.transpose(py_x.sample(num_of_samples)), (-1, 1)), tf.float32)
    for layer in range(0,len(ts_layers)):
        ts = tf.cast(tf.repeat(ts_layers[layer], num_of_samples, axis=0), tf.float32)
        ixt =train_mine(xs, ts_layers[layer], batch_size=batch_size, epochs=epochs)
        ity =train_mine(ts, ys, batch_size=batch_size, epochs=epochs)
        information.append([ixt, ity])
        print (layer, ixt, ity)
    return information


class MINE(Model):
    """The MINE estimator for estimate MI """
    def __init__(self, network=None, hidden_dim = 100,layers=3, activation='elu'):
        super(MINE, self).__init__()
        if not network:
            self.network =  mlp(hidden_dim, 1, layers, activation)
        else:
            self.network = network

    def call(self, inputs):
        x, y = inputs['x'], inputs['y']
        y_bar_input = tf.random.shuffle(y)  # shuffle y input as y_bar
        ing = tf.concat([x, y], axis=1)
        t_xy = self.network(ing)
        ingf = tf.concat([x, y_bar_input], axis=1)
        t_xy_bar = self.network(ingf)
        loss = mine_loss(t_xy-1, t_xy_bar-1)
        self.add_loss(loss)
        return tf.zeros(1,1)