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
def loss_func_ib(hyz, hzy, hzx, hyhatz, hyz_noisy, log_beta, gamma, noisy_learning=False):
    info_loss = hzy - hzx
    if noisy_learning:
        label_term = hyz_noisy
    else:
        label_term = hyz
    total_loss = info_loss + log_beta * label_term
    return total_loss


@tf.function
def loss_func_combined(hyz, hzy, hzx, hyhatz, hyz_noisy, log_beta, gamma, noisy_learning=False):
    info_loss = hzy - hzx
    if noisy_learning:
        label_term = hyz_noisy
    else:
        label_term = hyz
    total_loss = info_loss + log_beta * label_term + gamma * hyhatz
    return total_loss


@tf.function
def loss_func_dual_ib(hyz, hzy, hzx, hyhatz, hyz_noisy, log_beta, gamma, noisy_learning=False):
    info_loss = hzy - hzx
    total_loss = info_loss + log_beta * hyhatz
    return total_loss


def loss_func(labels, labels_dist, logits, z, encoding, prior, log_beta, num_of_labels=10, scale=.1,
              loss_func_inner=None, gamma=0,
              confusion_matrix=None, eps=1e-6, use_logprob_func=True,
              label_smoothing=0.2, label_noise_type='confusion_matrix_noise',
              labels_noise=0.1, x=None, pre_model=None):
    label_onehot = tf.one_hot(labels, num_of_labels)
    # cyz = tfd.Categorical(logits=logits)
    if label_noise_type == 'confusion_matrix_noise':
        # tf.print (confusion_matrix.shape,labels.shape )
        label_onehot_with_noise = tf.cast(tf.gather(confusion_matrix, tf.cast(labels, tf.int32)), tf.float32)

        label_onehot_with_noise = tf.clip_by_value(label_onehot_with_noise, eps, 1 - eps)

    elif label_noise_type == 'gaussian_noise':
        label_onehot_with_noise_before = tf.random.normal(label_onehot.shape, mean=label_onehot, stddev=labels_noise)
        label_onehot_with_noise_before = label_onehot_with_noise_before + eps
        label_onehot_with_noise = label_onehot_with_noise_before / tf.reduce_sum(label_onehot_with_noise_before,
                                                                                 axis=1)[:, None]
        label_onehot_with_noise = tf.clip_by_value(label_onehot_with_noise, eps, 1 - eps)
        # print (label_onehot_with_noise[0], label_onehot[0])
    elif label_noise_type == 'smooth_noise':
        label_onehot_with_noise = (label_onehot * (1.0 - label_smoothing) + (label_smoothing / num_of_labels))
    elif label_noise_type == 'pre_defined_model':
        label_onehot_with_noise_before = pre_model(x)
        label_onehot_with_noise = tf.math.softmax(label_onehot_with_noise_before)
        label_onehot_with_noise = tf.clip_by_value(label_onehot_with_noise, eps, 1 - eps)
        # print (label_onehot_with_noise[0], label_onehot_with_noise_before[0], label_onehot[0])

    # hyz = -cyz.log_prob(tf.cast(labels, tf.int32))

    hyz = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.0,
                                                  reduction=tf.keras.losses.Reduction.NONE)(y_pred=logits,
                                                                                            y_true=label_onehot)
    hyz_noisy = tf.keras.losses.categorical_crossentropy(from_logits=True, label_smoothing=0.0, y_pred=logits,
                                                         y_true=label_onehot_with_noise)

    if use_logprob_func:
        tf.print('NNNNNNNN')
        if labels_dist == None:
            # batch_siz_e = labels_noise * tf.cast(tf.ones_like(label_onehot), tf.float32)
            labels_dist = tfd.Normal(loc=tf.cast(label_onehot, tf.float32),
                                     std=labels_noise)
        logits_model = tfd.Categorical(logits)
        hyhatz = logits_model.entropy() - (labels_dist.log_prob(logits))
    else:
        logits_probs = tf.nn.softmax(logits)
        logits_probs = tf.clip_by_value(logits_probs, eps, 1 - eps)

        # cliped_y_pref_tf = tf.clip_by_value(label_onehot_with_noise, eps, 1 - eps)

        hyhatz = tf.reduce_sum(logits_probs * tf.math.log(logits_probs), axis=1) - tf.reduce_sum(
            logits_probs * tf.math.log(label_onehot_with_noise), axis=1)

    hzx = -encoding.log_prob(z)
    hzy = -prior.log_prob(z)
    total_loss = loss_func_inner(hyz=hyz, hzy=hzy, hzx=hzx, hyhatz=hyhatz,
                                 hyz_noisy=hyz_noisy, log_beta=log_beta, gamma=gamma)
    total_loss = tf.where(tf.math.is_nan(total_loss), tf.zeros_like(total_loss), total_loss)
    total_loss = tf.reduce_mean(total_loss)
    losses = [hzy, hzx, hyz, hyhatz, hyz_noisy]
    return total_loss, losses


class VariationalNetwork(keras.Model):
    def __init__(self, log_beta, gamma, labels_dist, encoder, decoder, prior, loss_func_inner,
                 beta_sched, gamma_sched, labels_noise=1e-2,
                 confusion_matrix=None, use_logprob_func=True,
                 file_name=None, label_noise_type='gaussian_noise', pre_model=None, model_path=None):
        super(VariationalNetwork, self).__init__()
        self.log_beta = log_beta
        self.gamma = gamma
        self.loss_func_inner = loss_func_inner
        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior
        self.beta_sched = beta_sched
        self.gamma_sched = gamma_sched
        self.labels_noise = labels_noise
        self.confusion_matrix = confusion_matrix
        self.step = 0
        self.labels_dist = labels_dist
        self.file_name = file_name
        self.labels_noise = labels_noise
        self.label_noise_type = label_noise_type
        self.use_logprob_func = use_logprob_func
        self.pre_model = pre_model
        self.model_path = model_path

    def call(self, x):
        params = self.encoder(x)
        mu, rho = params[:, :self.encoder.z_dim], params[:, self.encoder.z_dim:]

        encoding = tfp.layers.DistributionLambda(
            lambda t: tfd.MultivariateNormalDiag(loc=t[0], scale_diag=1e-3 + tf.math.softplus(t[1])))([mu, rho])

        z = encoding.sample()
        cyz = self.decoder(z)
        return encoding, cyz, z

    def write_metrices(self, y, logits, losses, total_loss, beta, gamma, num_of_labels=10, step=0, mode=None):
        hzy, hzx, hyz, hyhatz, hyz_noisy = losses
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
                if self.metrics[i].name == 'hyz_noisy':
                    self.metrics[i].update_state(hyz_noisy)
                if self.metrics[i].name == 'beta':
                    self.metrics[i].update_state(beta)
        self.compiled_loss(
            tf.one_hot(y, num_of_labels), logits)

    def test_step(self, data):
        x, y = data
        # Compute predictions
        encoding, cyz, z = self(x)
        prior = self.prior(y)
        log_beta = self.beta_sched(self.optimizer.iterations)
        gamma = self.gamma_sched(self.optimizer.iterations)

        labels_dist = self.labels_dist(y)
        # if self.model_path:

        # self.pre_model.load_weights(model_path)

        total_loss, losses = loss_func(y, labels_dist, cyz, z, encoding, prior, log_beta,
                                       loss_func_inner=self.loss_func_inner, scale=self.labels_noise,
                                       confusion_matrix=self.confusion_matrix, use_logprob_func=self.use_logprob_func,
                                       label_noise_type=self.label_noise_type, gamma=gamma,
                                       labels_noise=self.labels_noise, pre_model=self.pre_model, x=x)
        z = encoding.mean()
        cyz = self.decoder(z)
        self.write_metrices(y, cyz, losses, total_loss, log_beta, step=self.optimizer.iterations,
                            gamma=gamma, mode='test')
        return {m.name: m.result() for m in self.metrics}

    def train_step(self, input):
        x, y = input
        # if self.model_path and (self.step % 5) ==0:
        #  self.pre_model.load_weights(self.model_path)

        # print ('sssss', self.pre_model.evaluate(x,y, steps = 5))
        with tf.GradientTape() as tape:
            self.step = self.step + 1
            encoding, cyz, z = self(x)
            prior = self.prior(y)
            log_beta = self.beta_sched(self.optimizer.iterations)
            gamma = self.gamma_sched(self.optimizer.iterations)
            labels_dist = self.labels_dist(y)
            # labels_dist = []
            total_loss, losses = loss_func(y, labels_dist, cyz, z, encoding,
                                           prior, log_beta, loss_func_inner=self.loss_func_inner,
                                           scale=self.labels_noise, gamma=gamma, use_logprob_func=self.use_logprob_func,
                                           confusion_matrix=self.confusion_matrix,
                                           label_noise_type=self.label_noise_type, labels_noise=self.labels_noise,
                                           pre_model=self.pre_model, x=x)
        self.write_metrices(y, cyz, losses, total_loss, log_beta, step=self.optimizer.iterations, gamma=gamma,
                            mode='train')
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {m.name: m.result() for m in self.metrics}
