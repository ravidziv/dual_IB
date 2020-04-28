import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
tfd = tfp.distributions


#@tf.function
def loss_func_ib(hyz, hzy, hzx, hyhatz, log_beta):
    info_loss =hzy - hzx
    total_loss = hyz + tf.exp(-log_beta) * info_loss
    return total_loss


#@tf.function
def loss_func_dual_ib(hyz, hzy, hzx, hyhatz, log_beta):
    info_loss =hzy - hzx
    total_loss = hyhatz - tf.exp(-log_beta) * info_loss
    return total_loss

#@tf.function
def loss_func(labels, logits, z, encoding, prior, log_beta, num_of_labels=10, scale= 1.5, loss_func_inner=None):
    cyz = tfd.Categorical(logits=logits)
    labels_one_hot = tf.one_hot(labels, num_of_labels)
    #labels_one_hot-= 1./num_of_labels
    batch_siz_e = scale*tf.cast(tf.ones_like(labels_one_hot), tf.float32)

    labels_dist = tfd.MultivariateNormalDiag(loc=tf.cast(labels_one_hot, tf.float32),
                                             scale_diag=batch_siz_e)
    hyhatz = -labels_dist.log_prob(logits)
    hyz =-cyz.log_prob(tf.cast(labels, tf.int32))
    hzx = -encoding.log_prob(z)
    hzy = -prior.log_prob(z)
    total_loss = tf.reduce_mean(loss_func_inner(hyz, hzy, hzx, hyhatz, log_beta))
    losses = [hzy,hzx ,hyz,hyhatz ]
    return total_loss, losses


class VariationalNetwork(keras.Model):
    def __init__(self, log_beta, encoder, decoder, prior, loss_func_inner, beta_sched, labels_noise = 1e-2):
        super(VariationalNetwork, self).__init__()
        self.log_beta = log_beta
        self.loss_func_inner=loss_func_inner
        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior
        self.beta_sched=beta_sched
        self.labels_noise = labels_noise
        self.step=0

    def call(self, x):
        encoding = self.encoder(x)
        z = encoding.sample()
        cyz = self.decoder(z)
        return encoding, cyz, z


    def write_metrices(self, y, logits, losses, total_loss, num_of_labels=10 ):
        hzy, hzx, hyz, hyhatz = losses
        if  len(self.metrics)>0:
            for i in range(len(self.metrics)):
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

        self.compiled_metrics.update_state(tf.one_hot(y, num_of_labels), logits)
        self.compiled_loss(
            tf.one_hot(y, num_of_labels), logits)
    def test_step(self, data):
        x, y = data
        # Compute predictions
        encoding, cyz, z = self(x)
        prior = self.prior(y)
        total_loss, losses = loss_func(y, cyz, z, encoding, prior, self.log_beta, loss_func_inner=self.loss_func_inner, scale = self.labels_noise)
        encoding = self.encoder(x)
        z = encoding.mean()
        cyz = self.decoder(z)
        self.write_metrices(y, cyz, losses, total_loss)
        return {m.name: m.result() for m in self.metrics}


    def train_step(self, input):
        x, y = input
        with tf.GradientTape() as tape:
            encoding, cyz, z = self(x)
            prior = self.prior(y)
            log_beta = self.beta_sched(self.log_beta, self.step)

            total_loss, losses = loss_func(y, cyz, z, encoding,
                                           prior, log_beta, loss_func_inner=self.loss_func_inner, scale = self.labels_noise)
        self.step+=1
        self.write_metrices(y, cyz, losses, total_loss)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {m.name: m.result() for m in self.metrics}

