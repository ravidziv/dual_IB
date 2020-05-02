import tensorflow as tf

def bulid_model(input_shape, y_dim=2, nonlin = 'tanh', layers_width=[]):
  model = tf.keras.Sequential()

  model.add(tf.keras.layers.Dense(layers_width[0], input_shape=(input_shape,), activation=nonlin))
  for i in range(1, len(layers_width)):
    model.add(tf.keras.layers.Dense(layers_width[i], activation=nonlin))
  model.add(tf.keras.layers.Dense(y_dim, activation='softmax'))
  return model
