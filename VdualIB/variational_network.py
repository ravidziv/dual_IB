import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp

tfd = tfp.distributions
import pandas as pd
from csv import writer


def save_pickle(file_name, list_of_elem):
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


@tf.function
def loss_func_ib(hyz, hzy, hzx, hyhatz, log_beta):
    info_loss = hzy - hzx
    total_loss = info_loss + log_beta * hyz
    return total_loss


@tf.function
def loss_func_dual_ib(hyz, hzy, hzx, hyhatz, log_beta):
    info_loss = hzy - hzx
    total_loss = info_loss + log_beta * hyhatz
    return total_loss


def loss_func(labels, labels_dist, logits, z, encoding, prior, log_beta, num_of_labels=10, scale=.1,
              loss_func_inner=None,
              confusion_matrix=None, eps=1e-6, use_logprob_func=True,
              GLOBAL_BATCH_SIZE=None):
    label_onehot = tf.one_hot(labels, num_of_labels)
    cyz = tfd.Categorical(logits=logits)
    hyz = -cyz.log_prob(tf.cast(labels, tf.int32))
    # hyz =tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.0,  reduction =tf.keras.losses.Reduction.NONE)(y_pred =logits, y_true=label_onehot)
    if use_logprob_func:
        if labels_dist == None:
            batch_siz_e = scale * tf.cast(tf.ones_like(label_onehot), tf.float32)

            labels_dist = tfd.MultivariateNormalDiag(loc=tf.cast(label_onehot, tf.float32),
                                                     scale_diag=batch_siz_e)
        hyhatz = -labels_dist.log_prob(logits)
    else:
        if confusion_matrix == None:
            label_onehot_with_noise = tf.cast(confusion_matrix[labels], tf.float32)
        else:
            label_onehot_with_noise = tf.where(label_onehot == 1, 0.99, label_onehot)
            label_onehot_with_noise = tf.where(label_onehot == 0, 0.01, label_onehot_with_noise)
        logits_probs = tf.nn.softmax(logits)
        logits_probs = tf.clip_by_value(logits_probs, eps, 1 - eps)
        cliped_y_pref_tf = tf.clip_by_value(label_onehot_with_noise, eps, 1 - eps)
        hyhatz = tf.reduce_sum(logits_probs * tf.math.log(logits_probs), axis=1) - tf.reduce_sum(
            logits_probs * tf.math.log(cliped_y_pref_tf), axis=1)
    hzx = -encoding.log_prob(z)
    hzy = -prior.log_prob(z)
    total_loss = loss_func_inner(hyz, hzy, hzx, hyhatz, log_beta)
    total_loss = tf.where(tf.math.is_nan(total_loss), tf.zeros_like(total_loss), total_loss)
    total_loss = tf.reduce_mean(total_loss)
    losses = [hzy, hzx, hyz, hyhatz]
    return total_loss, losses


class VariationalNetwork(keras.Model):
    def __init__(self, log_beta, labels_dist, encoder, decoder, prior, loss_func_inner, beta_sched, labels_noise=1e-2,
                 confusion_matrix=None,
                 file_name=None, measures_list=[]):
        super(VariationalNetwork, self).__init__()
        self.log_beta = log_beta
        self.loss_func_inner = loss_func_inner
        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior
        self.beta_sched = beta_sched
        self.labels_noise = labels_noise
        self.confusion_matrix = confusion_matrix
        self.step = 0
        self.labels_dist = labels_dist
        self.file_name = file_name
        # list_of_elem = ['step', 'loss', 'mode'].extend(measures_list)
        # self.df = pd.DataFrame(columns=[list_of_elem])
        # save_pickle(self.file_name, list_of_elem)

    def call(self, x):
        params = self.encoder(x)
        mu, rho = params[:, :self.encoder.z_dim], params[:, self.encoder.z_dim:]

        encoding = tfp.layers.DistributionLambda(
            lambda t: tfd.MultivariateNormalDiag(loc=t[0], scale_diag=1e-3 + tf.math.softplus(t[1])))([mu, rho])

        z = encoding.sample()
        cyz = self.decoder(z)
        return encoding, cyz, z

    def write_metrices(self, y, logits, losses, total_loss, beta, num_of_labels=10, step=0, mode=None):
        hzy, hzx, hyz, hyhatz = losses
        self.compiled_metrics.update_state(tf.one_hot(y, num_of_labels), logits)
        izy = tf.math.log(tf.cast(num_of_labels, tf.float32)) - hyz
        izyhat = tf.math.log(tf.cast(num_of_labels, tf.float32)) - hyhatz
        ixt = hzy - hzx
        if len(self.metrics) > 0:

            for i in range(len(self.metrics)):
                if self.metrics[i].name != 'accuracy':
                    # continue
                    self.metrics[i].reset_states()
                else:
                    acc = self.metrics[i].result
                if self.metrics[i].name == 'hzy':
                    self.metrics[i].update_state(hzy)
                if self.metrics[i].name == 'hzx':
                    self.metrics[i].update_state(hzx)
                if self.metrics[i].name == 'hyz':
                    self.metrics[i].update_state(hyz)
                if self.metrics[i].name == 'hyhatz':
                    self.metrics[i].update_state(hyhatz)
                if self.metrics[i].name == 'ixt':
                    self.metrics[i].update_state(ixt)
                if self.metrics[i].name == 'total_loss':
                    self.metrics[i].update_state(total_loss)
                if self.metrics[i].name == 'izy':
                    self.metrics[i].update_state(izy)
                if self.metrics[i].name == 'izyhat':
                    self.metrics[i].update_state(izyhat)
                if self.metrics[i].name == 'beta':
                    self.metrics[i].update_state(beta)
        self.compiled_loss(
            tf.one_hot(y, num_of_labels), logits)
        # dic_t = {m.name: m.result() for m in self.metrics}
        # dic_t['step'] = step
        # dic_t['mode'] = mode
        # df.append(dic_t)
        # df.to_csv(self.file_name)

    def test_step(self, data):
        x, y = data
        # Compute predictions
        encoding, cyz, z = self(x)
        prior = self.prior(y)
        log_beta = self.beta_sched(self.optimizer.iterations)
        labels_dist = self.labels_dist(y)

        total_loss, losses = loss_func(y, labels_dist, cyz, z, encoding, prior, log_beta,
                                       loss_func_inner=self.loss_func_inner, scale=self.labels_noise,
                                       confusion_matrix=self.confusion_matrix, GLOBAL_BATCH_SIZE=0)
        z = encoding.mean()
        cyz = self.decoder(z)
        self.write_metrices(y, cyz, losses, total_loss, log_beta, step=self.optimizer.iterations,
                            mode='test')
        return {m.name: m.result() for m in self.metrics}

    def train_step(self, input):
        x, y = input
        with tf.GradientTape() as tape:
            self.step = self.step + 1
            encoding, cyz, z = self(x)
            prior = self.prior(y)
            log_beta = self.beta_sched(self.optimizer.iterations)
            labels_dist = self.labels_dist(y)
            # labels_dist = []
            total_loss, losses = loss_func(y, labels_dist, cyz, z, encoding,
                                           prior, log_beta, loss_func_inner=self.loss_func_inner,
                                           scale=self.labels_noise,
                                           confusion_matrix=self.confusion_matrix, GLOBAL_BATCH_SIZE=0)
        self.write_metrices(y, cyz, losses, total_loss, log_beta, step=self.optimizer.iterations,
                            mode='train'
                            )
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {m.name: m.result() for m in self.metrics}