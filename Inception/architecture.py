""" Create Inception model architecture."""
import tensorflow as tf

# TODO: change to Inception.modules
import modules

def build_model():
    """ Build and return Tensorflow model with Inception architecure."""
    # Input layer.
    input_layer = tf.keras.layers.Input(shape=(299,299,3))

    # Stem module.
    stem = modules.InceptionStemBlock()(input_layer)

    # Inception modules.
    # Four inception blocks and reduction.
    inception_1a = modules.InceptionBlockA()(stem)
    inception_1b = modules.InceptionBlockA()(inception_1a)
    inception_1c = modules.InceptionBlockA()(inception_1b)
    inception_1d = modules.InceptionBlockA()(inception_1c)
    inception_1_reduced = modules.InceptionReductionA()(inception_1d)

    # Seven inception blocks and reductions.
    inception_2a = modules.InceptionBlockB()(inception_1_reduced)
    inception_2b = modules.InceptionBlockB()(inception_2a)
    inception_2c = modules.InceptionBlockB()(inception_2b)
    inception_2d = modules.InceptionBlockB()(inception_2c)
    inception_2e = modules.InceptionBlockB()(inception_2d)
    inception_2f = modules.InceptionBlockB()(inception_2e)
    inception_2g = modules.InceptionBlockB()(inception_2f)
    inception_2_reduced = modules.InceptionReductionB()(inception_2g)

    # Three inception blocks.
    inception_3a = modules.InceptionBlockC()(inception_2_reduced)
    inception_3b = modules.InceptionBlockC()(inception_3a)
    inception_3c = modules.InceptionBlockC()(inception_3b)

    # Average pooling and dropout layers.
    averagepool_layer = tf.keras.layers.AveragePooling2D(
        pool_size=(8,8),
        strides=(1,1),
        name='inception_averagepool'
    )(inception_3c)
    dropout_layer = tf.keras.layers.Dropout(rate=0.2)(averagepool_layer)

    # Output.
    output_layer = tf.keras.layers.Dense(
        units=1000,
        activation='softmax',
        name='output_layer'
    )(dropout_layer)

    inception_model = tf.keras.Model(inputs=stem, outputs=[output_layer])
    print(inception_model.summary())
    return inception_model

build_model()
