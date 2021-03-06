"""A variational networks - CEB, VIB and VdualIB """
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp

tfd = tfp.distributions


@tf.function
def loss_func_inner(hyz, hzy, hzx, hyhatz, hyz_noisy, beta, gamma, noisy_learning=False):
    """Loss function for the duablIB/CEB """
    info_loss = hzy - hzx
    if noisy_learning:
        label_term = hyz_noisy
    else:
        label_term = hyz
    total_loss = info_loss + beta * label_term + gamma * hyhatz
    return total_loss


def loss_func(labels, labels_dist_func, logits, z, encoding, prior, beta, num_of_labels=10, scale=.1,
              loss_func_inner=None, gamma=0, noisy_learning=True,
              confusion_matrix=None, eps=1e-7, use_logprob_func=True,
              label_smoothing=0.02, label_noise_type='confusion_matrix_noise',
              labels_noise=0.001, x=None, pre_model=None):
    """Loss function for variatioanl network"""
    label_onehot = tf.one_hot(labels, num_of_labels)
    # What noise to insert  to the labels
    if label_noise_type == 'confusion_matrix_noise':
        label_onehot_with_noise = tf.cast(tf.gather(confusion_matrix, tf.cast(labels, tf.int32)), tf.float32)
        label_onehot_with_noise = tf.clip_by_value(label_onehot_with_noise, eps, 1 - eps)
    elif label_noise_type == 'gaussian_noise':
        label_noise_un = tf.random.normal(label_onehot.shape, mean=label_onehot, stddev=labels_noise)
        label_noise_un = tf.clip_by_value(label_noise_un, eps, 1 - eps)
        label_onehot_with_noise = label_noise_un / tf.reduce_sum(label_noise_un, axis=1)[:, None]
    elif label_noise_type == 'smooth_noise':
        label_onehot_with_noise = (label_onehot * (1.0 - label_smoothing) + (label_smoothing / num_of_labels))
    elif label_noise_type == 'pre_defined_model':
        label_noise_un = pre_model(x)
        label_onehot_with_noise = tf.math.softmax(label_noise_un)
        label_onehot_with_noise = tf.clip_by_value(label_onehot_with_noise, eps, 1 - eps)
    # Calculate logprob function analytically or not
    if use_logprob_func:
        labels_dist = labels_dist_func(labels)
        logits_model = tfd.Categorical(logits)
        hyhatz = logits_model.entropy() - (labels_dist.log_prob(logits))
    else:
        logits_probs = tf.nn.softmax(logits)
        logits_probs = tf.clip_by_value(logits_probs, eps, 1 - eps)
        hyhatz = tf.reduce_sum(logits_probs * tf.math.log(logits_probs), axis=1) - tf.reduce_sum(
            logits_probs * tf.math.log(label_onehot_with_noise), axis=1)
    hyz = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.0,
                                                  reduction=tf.keras.losses.Reduction.NONE)(y_pred=logits,
                                                                                            y_true=label_onehot)
    hyz_noisy = tf.keras.losses.categorical_crossentropy(from_logits=True, label_smoothing=0.0, y_pred=logits,
                                                         y_true=label_onehot_with_noise)
    hzx = -encoding.log_prob(z)
    hzy = -prior.log_prob(z)
    total_loss = loss_func_inner(hyz=hyz, hzy=hzy, hzx=hzx, hyhatz=hyhatz,
                                 hyz_noisy=hyz_noisy, beta=beta, gamma=gamma, noisy_learning=noisy_learning)
    total_loss = tf.reduce_mean(total_loss)
    losses = [hzy, hzx, hyz, hyhatz, hyz_noisy]
    return total_loss, losses


class VariationalNetwork(keras.Model):
    def __init__(self, beta, gamma, labels_dist, encoder, decoder, prior, beta_sched, gamma_sched,
                 loss_func_inner=loss_func_inner,
                 labels_noise=1e-2,
                 confusion_matrix=None, use_logprob_func=True,
                 file_name=None, label_noise_type='gaussian_noise', pre_model=None, model_path=None,
                 num_of_labels=10, noisy_learning=True):
        super(VariationalNetwork, self).__init__()
        self.beta = beta
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
        self.num_of_labels = num_of_labels
        self.noisy_learning = noisy_learning

    def call(self, x):
        #Sample from multivariate gaussian around the output of the encoder and pass it to the decoder
        params = self.encoder(x)
        mu, rho = params[:, :self.encoder.z_dim], params[:, self.encoder.z_dim:]
        encoding = tfp.layers.DistributionLambda(
            lambda t: tfd.MultivariateNormalDiag(loc=t[0], scale_diag=1e-3 + tf.math.softplus(t[1])))([mu, rho])
        z = encoding.sample()
        cyz = self.decoder(z)
        return encoding, cyz, z

    def write_metrices(self, y, logits, losses, total_loss, beta, gamma, num_of_labels=10):
        hzy, hzx, hyz, hyhatz, hyz_noisy = losses
        self.compiled_metrics.update_state(tf.one_hot(y, num_of_labels), logits)
        izy = tf.math.log(tf.cast(num_of_labels, tf.float32)) - hyz
        izyhat = tf.math.log(tf.cast(num_of_labels, tf.float32)) - hyhatz
        ixt = hzy - hzx
        # The key values to store in the metrics
        dict_met_vals = {'hzy': hzy, 'hzx': hzx, 'hyz': hyz, 'hyhatz': hyhatz, 'ixt': ixt, 'total_loss': total_loss,
                         'izy': izy, 'izyhat': izyhat, 'hyz_noisy': hyz_noisy, 'beta': beta, 'gamma': gamma}
        # Insert values to the metrics
        for i in range(len(self.metrics)):
            metr = self.metrics[i]
            if metr.name in dict_met_vals:
                metr.update_state(dict_met_vals[metr.name])
        self.compiled_loss(
            tf.one_hot(y, num_of_labels), logits)

    def test_step(self, data):
        x, y = data
        # Compute predictions
        encoding, cyz, z = self(x)
        prior = self.prior(y)
        beta = self.beta_sched(self.optimizer.iterations)
        gamma = self.gamma_sched(self.optimizer.iterations)
        total_loss, losses = loss_func(labels=y, labels_dist_func=self.labels_dist, logits=cyz,
                                       z=z, encoding=encoding, prior=prior, beta=beta,
                                       loss_func_inner=self.loss_func_inner, scale=self.labels_noise,
                                       confusion_matrix=self.confusion_matrix, use_logprob_func=self.use_logprob_func,
                                       label_noise_type=self.label_noise_type, gamma=gamma,
                                       labels_noise=self.labels_noise, pre_model=self.pre_model, x=x,
                                       num_of_labels=self.num_of_labels, noisy_learning=self.noisy_learning)
        # In the test we calculate the predication based on the mean
        z = encoding.mean()
        cyz = self.decoder(z)
        self.write_metrices(y=y, logits=cyz, losses=losses, total_loss=total_loss, beta=beta,
                            gamma=gamma, num_of_labels=self.num_of_labels)
        return {m.name: m.result() for m in self.metrics}

    def train_step(self, input):
        x, y = input
        with tf.GradientTape() as tape:
            self.step += 1
            encoding, cyz, z = self(x)
            prior = self.prior(y)
            beta = self.beta_sched(self.optimizer.iterations)
            gamma = self.gamma_sched(self.optimizer.iterations)
            labels_dist = self.labels_dist(y)
            total_loss, losses = loss_func(y, labels_dist, cyz, z, encoding,
                                           prior, beta, loss_func_inner=self.loss_func_inner,
                                           scale=self.labels_noise, gamma=gamma, use_logprob_func=self.use_logprob_func,
                                           confusion_matrix=self.confusion_matrix, noisy_learning=self.noisy_learning,
                                           label_noise_type=self.label_noise_type, labels_noise=self.labels_noise,
                                           pre_model=self.pre_model, x=x, num_of_labels=self.num_of_labels)
        self.write_metrices(y=y, logits=cyz, losses=losses, total_loss=total_loss, beta=beta, gamma=gamma,
                            num_of_labels=self.num_of_labels)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {m.name: m.result() for m in self.metrics}
