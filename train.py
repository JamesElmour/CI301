import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.optimizers.schedules import *

@tf.function
def transform(hr, lr):
    q = random.random()

    if((q > 0.5 and q < 0.6)):
        hr = tf.image.flip_left_right(hr)
        lr = tf.image.flip_left_right(lr)

    if((q > 0.4 and q < 0.5)):
        hr = tf.image.flip_up_down(hr)
        lr = tf.image.flip_up_down(lr)

    if((q > 0.6 and q < 0.75) or q > 0.9):
        hr = tf.image.rot90(hr)
        hr = tf.image.rot90(hr)
        lr = tf.image.rot90(lr)
        lr = tf.image.rot90(lr)

    if((q > 0.75 and q < 0.9) or q > 0.9):
        hr = tf.image.adjust_contrast(hr, 2)
        lr = tf.image.adjust_contrast(lr, 2)

    return hr, lr


pls_metric = tf.keras.metrics.Mean()
dls_metric = tf.keras.metrics.Mean()
step = 1
steps = 100000

for hr, lr in train_ds.take(steps):
    step += 1

    hr, lr = transform(hr, lr)

    # Train for a step
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        low_res = tf.cast(lr, tf.float32)
        high_res = tf.cast(hr, tf.float32)

        generated = generator(low_res, training = True)
        hr_output = discriminator(high_res, training = True)
        sr_output = discriminator(generated, training = True)

        con_loss = content_loss(hr, sr)
        gen_loss = generator_loss(sr_output)
        perc_loss = con_loss + 0.001 * gen_loss
        disc_loss = discriminator_loss(hr_output, sr_output)

    gradients_of_generator = gen_tape.gradient(perc_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    pls_metric(perc_loss)
    dls_metric(disc_loss)

    if step % 50 == 0:
        print(f'{step}/{steps}, perceptual loss = {pls_metric.result():.4f}, discriminator loss = {dls_metric.result():.4f}')
        pls_metric.reset_states()
        dls_metric.reset_states()
    
    if step % 200 == 0:
        show_example()

    if step % 2000 == 0:
        save_models(step + 78000)