# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def discriminator_block(x_in, num_filters, strides=1):
    x = Conv2D(num_filters, kernel_size=3, strides=strides, padding='same')(x_in)
    x = BatchNormalization(momentum=0.8)(x)
    return LeakyReLU(alpha=0.2)(x)

def discriminator(num_filters=64):
    x_in = Input(shape=(960, 1280, 3))

    x = discriminator_block(x_in, num_filters)
    x = discriminator_block(x, num_filters, strides=2)

    x = discriminator_block(x, num_filters)
    x = discriminator_block(x, num_filters, strides=2)

    x = discriminator_block(x, num_filters)
    x = discriminator_block(x, num_filters, strides=2)

    x = discriminator_block(x, num_filters * 2)
    x = discriminator_block(x, num_filters * 2, strides=2)
    x = discriminator_block(x, num_filters * 2, strides=2)
    x = discriminator_block(x, num_filters * 2, strides=2)
    x = discriminator_block(x, num_filters)

    x = Flatten()(x)

    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Dense(1, activation='sigmoid')(x)

    return Model(x_in, x)