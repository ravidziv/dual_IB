import tensorflow as tf
from tensorflow.keras import layers, models


def bulid_model(input_shape, y_dim=2, nonlin='tanh', layers_width=[]):
  model = tf.keras.Sequential()

  model.add(tf.keras.layers.Dense(layers_width[0], input_shape=(input_shape,), activation=nonlin))
  for i in range(1, len(layers_width)):
    model.add(tf.keras.layers.Dense(layers_width[i], activation=nonlin))
  model.add(tf.keras.layers.Dense(y_dim, activation='softmax'))
  return model

def build_covnet():
  model = models.Sequential()
  model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.Flatten())
  model.add(layers.Dense(64, activation='relu'))
  model.add(layers.Dense(10))
  return model


def build_densnet(layers_width=[500, 100, 50], y_dim=10, nonlin='relu', input_shape=32 * 32 * 3):
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Flatten(input_shape=(32, 32, 3)))
  model.add(tf.keras.layers.Dense(layers_width[0], activation=nonlin))
  for i in range(1, len(layers_width)):
    model.add(tf.keras.layers.Dense(layers_width[i], activation=nonlin))
  model.add(tf.keras.layers.Dense(y_dim))
  return model
