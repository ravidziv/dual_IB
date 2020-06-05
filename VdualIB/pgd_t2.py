import tensorflow as tf
import numpy as np


# Helper function to preprocess the image so that it can be inputted in MobileNetV2
def preprocess(image):
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, (224, 224))
  image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
  image = image[None, ...]
  return image


def loss_func(pred, labels):
  label_mask = tf.one_hot(tf.cast(labels, tf.int32),
                          10,
                          on_value=1.0,
                          off_value=0.0,
                          dtype=tf.float32)
  loss = tf.nn.softmax_cross_entropy_with_logits(
    logits=pred, labels=labels)
  return tf.reduce_mean(loss)
  correct_logit = tf.reduce_sum(label_mask * pred, axis=1)
  wrong_logit = tf.reduce_max((1 - label_mask) * pred - 1e4 * label_mask, axis=1)
  loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
  return loss


def create_adversarial_pattern(input_image, input_label, model):
  # image = tf.Variable(input_image)
  image = input_image
  with tf.GradientTape() as tape:
    tape.watch(image)
    encoding, cyz, z = model(input_image)
    loss = loss_func(input_label, cyz)

  # Get the gradients of the loss w.r.t to the input image.
  gradient = tape.gradient(loss, image)
  # Get the sign of the gradients to create the perturbation
  signed_grad = tf.sign(gradient)

  return signed_grad


def create_adver_multi(x_nat, input_label, model, num_steps=10, step_size=2, epsilon=8):
  x = x_nat
  for i in range(num_steps):
    signed = create_adversarial_pattern(x, input_label, model)
    x = np.add(x, step_size * signed, out=x, casting='unsafe')
    x = np.clip(x, x_nat - epsilon, x_nat + epsilon)
    x = np.clip(x, 0, 255)  # ensure valid pixel range
  return x


epsilons = [0, 0.01, 0.1, 0.15]
descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input')
                for eps in epsilons]


def get_adversarial_acc_metric(model, img_input, fgsm, fgsm_params):
  def adv_acc(y, _):
    # Generate adversarial examples

    # x_adv = fgsm.generate(img_input, **fgsm_params)
    # Consider the attack to be constant
    # x_adv = tf.stop_gradient(x_adv)

    # Accuracy on the adversarial examples
    x_adv = create_adver_multi(img_input, y, model, num_steps=10, step_size=1, epsilon=1e-3)
    encoding, cyz, z = model(x_adv)

    return tf.keras.metrics.categorical_accuracy(y, cyz)

  return adv_acc
