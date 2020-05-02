import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
tfd = tfp.distributions


@tf.function
def loss_func_ib(hyz, hzy, hzx, hyhatz, log_beta):
    info_loss =hzy - hzx
    total_loss = hyz + tf.exp(-log_beta) * info_loss
    return total_loss


@tf.function
def loss_func_dual_ib(hyz, hzy, hzx, hyhatz, log_beta):
    info_loss =hzy - hzx
    total_loss = hyhatz - tf.exp(-log_beta) * info_loss
    return total_loss


def loss_func(labels, labels_dist, logits, z, encoding, prior, log_beta, num_of_labels=10, scale=.1,
              loss_func_inner=None,
              confusion_matrix=None, eps=1e-6, use_logprob_func=True):
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
    total_loss = tf.reduce_mean(loss_func_inner(hyz, hzy, hzx, hyhatz, log_beta))
    losses = [hzy,hzx ,hyz,hyhatz ]
    return total_loss, losses


class VariationalNetwork(keras.Model):
    def __init__(self, log_beta, labels_dist, encoder, decoder, prior, loss_func_inner, beta_sched, labels_noise=1e-2,
                 confusion_matrix=None):
        super(VariationalNetwork, self).__init__()
        self.log_beta = log_beta
        self.loss_func_inner=loss_func_inner
        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior
        self.beta_sched=beta_sched
        self.labels_noise = labels_noise
        self.confusion_matrix = confusion_matrix
        self.step=0
        self.labels_dist = labels_dist

    def call(self, x):
        encoding = self.encoder(x)
        z = encoding.sample()
        cyz = self.decoder(z)
        return encoding, cyz, z

    def write_metrices(self, y, logits, losses, total_loss, beta, num_of_labels=10):
        hzy, hzx, hyz, hyhatz = losses
        self.compiled_metrics.update_state(tf.one_hot(y, num_of_labels), logits)

        if  len(self.metrics)>0:

            for i in range(len(self.metrics)):
                if self.metrics[i].name != 'accuracy':
                    # continue
                    self.metrics[i].reset_states()
                if  self.metrics[i].name == 'hzy':
                    self.metrics[i].update_state(hzy)
                if  self.metrics[i].name == 'hzx':
                    self.metrics[i].update_state(hzx)
                if self.metrics[i].name == 'hyz':
                    self.metrics[i].update_state(hyz)
                if self.metrics[i].name == 'hyhatz':
                    self.metrics[i].update_state(hyhatz)
                if self.metrics[i].name == 'ixt':
                    self.metrics[i].update_state(hzy - hzx)
                if self.metrics[i].name == 'total_loss':
                    self.metrics[i].update_state(total_loss)
                if self.metrics[i].name == 'izy':
                    self.metrics[i].update_state(tf.math.log(tf.cast(num_of_labels, tf.float32)) - hyz)
                if self.metrics[i].name == 'izyhat':
                    self.metrics[i].update_state(tf.math.log(tf.cast(num_of_labels, tf.float32)) - hyhatz)
                if self.metrics[i].name == 'beta':
                    self.metrics[i].update_state(beta)
        self.compiled_loss(
            tf.one_hot(y, num_of_labels), logits)

    def test_step(self, data):
        x, y = data
        # Compute predictions
        encoding, cyz, z = self(x)
        prior = self.prior(y)
        log_beta = self.beta_sched(self.log_beta, self.step)
        labels_dist = self.labels_dist(y)

        total_loss, losses = loss_func(y, labels_dist, cyz, z, encoding, prior, log_beta,
                                       loss_func_inner=self.loss_func_inner, scale=self.labels_noise,
                                       confusion_matrix=self.confusion_matrix)
        encoding = self.encoder(x)
        z = encoding.mean()
        cyz = self.decoder(z)
        self.write_metrices(y, cyz, losses, total_loss, log_beta)
        return {m.name: m.result() for m in self.metrics}


    def train_step(self, input):
        x, y = input
        with tf.GradientTape() as tape:
            encoding, cyz, z = self(x)
            prior = self.prior(y)
            log_beta = self.beta_sched(self.log_beta, self.step)
            labels_dist = self.labels_dist(y)
            # labels_dist = []
            total_loss, losses = loss_func(y, labels_dist, cyz, z, encoding,
                                           prior, log_beta, loss_func_inner=self.loss_func_inner,
                                           scale=self.labels_noise,
                                           confusion_matrix=self.confusion_matrix)
        self.step+= 1
        self.write_metrices(y, cyz, losses, total_loss, log_beta)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {m.name: m.result() for m in self.metrics}

