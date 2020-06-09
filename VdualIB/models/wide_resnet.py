from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, Dense, Input, add, Activation, Flatten, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

IN_FILTERS = 16


def wide_residual_network(input_shape, classes_num, depth, k, weight_decay, include_top=False,
                          is_cifar=True, with_batchnorm=True):
    img_input = Input(shape=input_shape)

    n_filters = [16, 16 * k, 32 * k, 64 * k]
    n_stack = (depth - 4) // 6

    def conv3x3(x, filters):
        return Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(weight_decay),
                      use_bias=False)(x)

    def bn_relu(x):
        if with_batchnorm:
            x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = Activation('relu')(x)
        return x

    def residual_block(x, out_filters, increase=False, dropout_rate=0.3):
        global IN_FILTERS
        stride = (1, 1)
        if increase:
            stride = (2, 2)

        o1 = bn_relu(x)

        conv_1 = Conv2D(out_filters,
                        kernel_size=(3, 3), strides=stride, padding='same',
                        kernel_initializer='he_normal',
                        kernel_regularizer=l2(weight_decay),
                        use_bias=False)(o1)
        # conv_1 = Dropout(dropout_rate)(conv_1)
        o2 = bn_relu(conv_1)

        conv_2 = Conv2D(out_filters,
                        kernel_size=(3, 3), strides=(1, 1), padding='same',
                        kernel_initializer='he_normal',
                        kernel_regularizer=l2(weight_decay),
                        use_bias=False)(o2)
        if increase or IN_FILTERS != out_filters:
            proj = Conv2D(out_filters,
                          kernel_size=(1, 1), strides=stride, padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=l2(weight_decay),
                          use_bias=False)(o1)
            block = add([conv_2, proj])
        else:
            block = add([conv_2, x])
        return block

    def wide_residual_layer(x, out_filters, increase=False):
        global IN_FILTERS
        x = residual_block(x, out_filters, increase)
        IN_FILTERS = out_filters
        for _ in range(1, int(n_stack)):
            x = residual_block(x, out_filters)
        return x

    x = conv3x3(img_input, n_filters[0])
    x = wide_residual_layer(x, n_filters[1])
    x = wide_residual_layer(x, n_filters[2], increase=True)
    x = wide_residual_layer(x, n_filters[3], increase=True)
    if with_batchnorm:
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    if is_cifar:
        x = AveragePooling2D((8, 8))(x)
    else:
        x = AveragePooling2D((7, 7))(x)
    x = Flatten()(x)
    if include_top:
        x = Dense(classes_num,
                  activation='linear',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(weight_decay),
                  use_bias=False)(x)
    model = Model(img_input, x)
    return model
