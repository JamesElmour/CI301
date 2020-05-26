# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x

import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

# Define pixel shuffle
def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)

# Create EDSR
def edsr(num_filters=64, num_res_blocks=16):
    x_in = Input(shape=(240, 320, 3))
    x = b = Conv2D(num_filters, 3, padding='same')(x_in)
    
    # Create residual blocks
    for i in range(num_res_blocks):
        r_in = x
        x = Conv2D(num_filters, 3, padding='same', activation='relu')(r_in)
        x = BatchNormalization()
        x = Conv2D(num_filters, 3, padding='same')(x)
        x = Add()([r_in, x])

    b = Conv2D(num_filters, 3, padding='same')(b)
    x = Add()([x, b])

    # Use pixel shuffle to upscale output 4x.
    x = Conv2D(num_filters * 4, 3, padding='same')(x)
    x = Lambda(pixel_shuffle(scale=2))(x)
    
    x = Conv2D(num_filters * 4, 3, padding='same')(x)
    x = Lambda(pixel_shuffle(scale=2))(x)

    x = Conv2D(3, 3, padding='same')(x)

    return Model(x_in, x, name="edsr")

def create_edsr():
    print("Loading model")
    model = edsr()
    model.summary()
    model.load_weights(os.getcwd() + "\generator.h5")

    return model