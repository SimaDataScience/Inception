""" Create Inception model architecture."""
import tensorflow as tf

# TODO: change to Inception.modules
import modules

def build_model():
    """ Build and return Tensorflow model with Inception architecure."""
    input_layer = tf.keras.layers.Input(shape=(299,299,3))

    conv_1a = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3,3),
        strides=(2,2),
        activation='relu',
        name='conv_1a'
    )(input_layer)
    conv_1b = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3,3),
        strides=(1,1),
        activation='relu',
        name='conv_1b'
    )(conv_1a)

    conv_zero_padded = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3,3),
        padding='same',
        strides=(1,1),
        activation='relu',
        name='conv_padded'
    )(conv_1b)
    conv_zero_padded_maxpool = tf.keras.layers.MaxPooling2D(
        pool_size=(3,3),
        strides=(2,2),
        name='conv_padded_maxpool'
    )(conv_zero_padded)

    conv_2a = tf.keras.layers.Conv2D(
        filters=80,
        kernel_size=(3,3),
        strides=(1,1),
        activation='relu',
        name='conv_2a'
    )(conv_zero_padded_maxpool)
    conv_2b = tf.keras.layers.Conv2D(
        filters=192,
        kernel_size=(3,3),
        strides=(2,2),
        activation='relu',
        name='conv_2b'
    )(conv_2a)
    conv_2c = tf.keras.layers.Conv2D(
        filters=192,
        kernel_size=(3,3),
        strides=(1,1),
        activation='relu',
        name='conv_2c'
    )(conv_2b)

    # Inception modules.
    # Three inception blocks and reduction.
    inception_1a = modules.InceptionBlockA(
        filters=,
        reductions=
    )(conv_2c)
    inception_1b = modules.InceptionBlockA(
        filters=,
        reductions=
    )(inception_1a)
    inception_1c = modules.InceptionBlockA(
        filters=,
        reductions=
    )(inception_1b)
    inception_1_reduced = modules.InceptionReductionA(
        filters=,
        reductions=
    )(inception_1c)

    # Five inception blocks and reductions.
    inception_2a = modules.InceptionBlockB(
        filters=,
        reductions=
    )(inception_1_reduced)
    inception_2b = modules.InceptionBlockB(
        filters=,
        reductions=
    )(inception_2a)
    inception_2c = modules.InceptionBlockB(
        filters=,
        reductions=
    )(inception_2b)
    inception_2d = modules.InceptionBlockB(
        filters=,
        reductions=
    )(inception_2c)
    inception_2e = modules.InceptionBlockB(

    )(inception_2d)
    inception_2_reduced = modules.InceptionReductionB(
        filters=,
        reductions=
    )(inception_2e)

    # Two inception layers.
    inception_3a = modules.InceptionBlockC(
        filters=,
        reductions=
    )(inception_2_reduced)
    inception_3b = modules.InceptionBlockC(
        filters=,
        reductions=
    )(inception_3a)

    inception_maxpool = tf.keras.layers.MaxPooling2D(
        pool_size=(8,8),
        strides=(1,1),
        name='inception_maxpool'
    )(inception_3b)

    linear_layer = tf.keras.layers.Dense(
        units=2048,
        activaton='relu',
        name='linear_layer'
    )(inception_maxpool)

    output_layer = tf.keras.layers.Dense(
        units=1000,
        activation='softmax',
        name='output_layer'
    )(linear_layer)

    inception_model = tf.keras.Model(inputs=input_layer, outputs=[output_layer])
    print(inception_model.summary())
    return inception_model
