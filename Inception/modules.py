""" Inception blocks."""
import tensorflow as tf

class ConvolutionalBlock(tf.keras.Model):
    """ 2D convolutional layer with batch normalization and relu activaiton."""
    def __init__(self, **kwargs):
        """ Initialize layers."""
        super(ConvolutionalBlock, self).__init__(name='')
        self.conv_layer = tf.keras.layers.Conv2D(**kwargs)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.Activation('relu')

    def call(self, input_tensor):
        """ Perform forward pass."""
        x_conv = self.conv_layer(input_tensor)
        x_normalized = self.batch_norm(x_conv)
        x_output = self.activation(x_normalized)

        return x_output

class InceptionBlockA(tf.keras.Model):
    """ Traditional inception block."""
    def __init__(self):
        """ Initialize layers."""
        super(InceptionBlockA, self).__init__(name='')
        # filter_1, filter_3, filter_5 = filters
        # reduction_3_1, reduction_3_2, reduction_maxpool = reductions

        self.conv_3a1 = ConvolutionalBlock(
            filters=64,
            kernel_size=(1,1)
        )
        self.conv_3a2 = ConvolutionalBlock(
            filters=96,
            kernel_size=(3,3)
        )

        # Two 3x3 convolutions after reduction, instead of 5x5 layer from original paper.
        self.conv_3b1 = ConvolutionalBlock(
            filters=64,
            kernel_size=(1,1)
        )
        self.conv_3b2 = ConvolutionalBlock(
            filters=96,
            kernel_size=(3,3)
        )
        self.conv_3b3 = ConvolutionalBlock(
            filters=96,
            kernel_size=(3,3)
        )

        self.conv_averagepool_a = tf.keras.layers.AveragePooling2D(
            pool_size=(3,3)
        )
        self.conv_averagepool_b = ConvolutionalBlock(
            filters=96,
            kernel_size=(1,1)
        )

        self.conv_1 = ConvolutionalBlock(
            filters=96,
            kernel_size=(1,1)
        )

        self.output_layer = tf.keras.layers.Concatenate()

    def call(self, input_tensor):
        """ Perform forward pass."""
        x_1 = self.conv_1(input_tensor)

        x_3a = self.conv_3a1(input_tensor)
        x_3a = self.conv_3a2(x_3a)

        x_3b = self.conv_3b1(input_tensor)
        x_3b = self.conv_3b2(x_3b)
        x_3b = self.conv_3b3(x_3b)

        x_averagepool = self.conv_averagepool_a(input_tensor)
        x_averagepool = self.conv_averagepool_b(x_averagepool)

        x_output = self.output_layer([ x_1, x_3a, x_3b, x_averagepool ])

        return x_output

class InceptionBlockB(tf.keras.Model):
    """ Computationally efficient factorization of nxn convolutional layer."""
    def __init__(self):
        """ Initialize layers."""
        super(InceptionBlockB, self).__init__(name='')
        # filter_1, filter_split_1, filter_split_n = filters
        # reduction_split_1, reduction_split_n, reduction_maxpool = reductions

        self.conv_deep1 = ConvolutionalBlock(
            filters=192,
            kernel_size=(1,1)
        )
        self.conv_deep2 = ConvolutionalBlock(
            filters=192,
            kernel_size=(1,7)
        )
        self.conv_deep3 = ConvolutionalBlock(
            filters=224,
            kernel_size=(7,1)
        )
        self.conv_deep4 = ConvolutionalBlock(
            filters=224,
            kernel_size=(1,7)
        )
        self.conv_deep5 = ConvolutionalBlock(
            filters=256,
            kernel_size=(7,1)
        )

        self.conv_shallow1 = ConvolutionalBlock(
            filters=192,
            kernel_size=(1,1)
        )
        self.conv_shallow2 = ConvolutionalBlock(
            filters=224,
            kernel_size=(1,7)
        )
        self.conv_shallow3 = ConvolutionalBlock(
            filters=256,
            kernel_size=(7,1)
        )

        self.conv_averagepool_a = tf.keras.layers.AveragePooling2D(
            pool_size=(3,3)
        )
        self.conv_averagepool_b = ConvolutionalBlock(
            filters=128,
            kernel_size=(1,1)
        )

        self.conv_1 = ConvolutionalBlock(
            filters=384,
            kernel_size=(1,1)
        )

        self.output_layer = tf.keras.layers.Concatenate()

    def call(self, input_tensor):
        """ Perform forward pass."""
        x_deep = self.conv_deep1(input_tensor)
        x_deep = self.conv_deep2(x_deep)
        x_deep = self.conv_deep3(x_deep)
        x_deep = self.conv_deep4(x_deep)
        x_deep = self.conv_deep5(x_deep)

        x_shallow = self.conv_shallow1(input_tensor)
        x_shallow = self.conv_shallow2(x_shallow)
        x_shallow = self.conv_n_short3(x_shallow)

        x_averagepool = self.conv_averagepool_a(input_tensor)
        x_averagepool = self.conv_averageool_b(x_averagepool)

        x_1 = self.conv_1(input_tensor)

        x_output = self.output_layer([x_deep, x_shallow, x_averagepool, x_1])

        return x_output

class InceptionBlockC(tf.keras.Model):
    """ Inception module with expanded filter bank."""
    def __init__(self):
        """ Initialize layers."""
        super(InceptionBlockC, self).__init__(name='')
        # filter_1, filter_split_1, filter_split_3a, filter_split_3b = filters
        # reduction_split_1, reduction_split_3, reduction_maxpool = reductions

        self.branch0_a = ConvolutionalBlock(
            filters=384,
            kernel_size=(1,1)
        )
        self.branch0_b = ConvolutionalBlock(
            filters=448,
            kernel_size=(1,3)
        )
        self.branch0_c = ConvolutionalBlock(
            filters=512,
            kernel_size=(3,1)
        )
        self.branch0a = ConvolutionalBlock(
            filters=256,
            kernel_size=(3,1)
        )
        self.branch0b = ConvolutionalBlock(
            filters=256,
            kernel_size=(1,3)
        )

        self.branch1_a = ConvolutionalBlock(
            filters=384,
            kernel_size=(1,1)
        )
        self.branch1a = ConvolutionalBlock(
            filters=256,
            kernel_size=(1,3)
        )
        self.branch1b = ConvolutionalBlock(
            filters=256,
            kernel_size=(3,1)
        )

        self.branch2_a = tf.keras.layers.AveragePooling2D(
            pool_size=(3,3)
        )
        self.branch2_b = ConvolutionalBlock(
            filters=256,
            kernel_size=(1,1)
        )

        self.branch3 = ConvolutionalBlock(
            filters=256,
            kernel_size=(1,1)
        )

        self.output_layer = tf.keras.layers.Concatenate()

    def call(self, input_tensor):
        """ Perform forward pass."""
        x_branch0 = self.branch0_a(input_tensor)
        x_branch0 = self.branch0_b(x_branch0)
        x_branch0 = self.branch0_c(x_branch0)

        x_branch0a = self.branch0a(x_branch0)
        x_branch0b = self.branch0b(x_branch0)

        x_branch1 = self.branch1_a(input_tensor)

        x_branch1a = self.branch1a(x_branch1)
        x_branch1b = self.branch1b(x_branch1)

        x_branch2 = self.branch2_a(input_tensor)
        x_branch2 = self.branch2_b(x_branch2)

        x_branch3 = self.branch3(input_tensor)

        x_output = self.output_layer(
            [
                x_branch0a, x_branch0b,
                x_branch1a, x_branch1b,
                x_branch2,
                x_branch3
            ]
        )

        return x_output

class InceptionReductionA(tf.keras.Model):
    """ Dimension reduction block."""
    def __init__(self):
        """ Initialize layers."""
        super(InceptionReductionA, self).__init__(name='')

        self.branch0_1 = ConvolutionalBlock(
            filters=192,
            kernel_size=(1,1)
        )
        self.branch0_2 = ConvolutionalBlock(
            filters=224,
            kernel_size=(3,3),
            strides=(1,1)
        )
        self.branch0_3 = ConvolutionalBlock(
            filters=256,
            kernel_size=(3,3),
            strides=(2,2),
            padding='same'
        )

        self.branch1 = ConvolutionalBlock(
            filters=384,
            kernel_size=(3,3),
            strides=(2,2),
            padding='same'
        )

        self.branch2 = tf.keras.layers.MaxPooling2D(
            pool_size=(3,3),
            strides=(2,2),
            padding='same'
        )

        self.output_layer = tf.keras.layers.Concatenate()

    def call(self, input_tensor):
        """ Perform forward pass."""
        x_branch0 = self.branch0_1(input_tensor)
        x_branch0 = self.branch0_2(input_tensor)
        x_branch0 = self.branch0_3(input_tensor)

        x_branch1 = self.branch1(input_tensor)

        x_branch2 = self.branch2(input_tensor)

        x_output = self.output_layer([x_branch0, x_branch1, x_branch2])

        return x_output

class InceptionReductionB(tf.keras.Model):
    """ Dimension reducton block."""
    def __init__(self):
        """ Initialize layers."""
        super(InceptionReductionB, self).__init__(name='')

        self.branch0_1 = ConvolutionalBlock(
            filters=256,
            kernel_size=(1,1)
        )
        self.branch0_2 = ConvolutionalBlock(
            filters=256,
            kernel_size=(1,7)
        )
        self.branch0_3 = ConvolutionalBlock(
            filters=320,
            kernel_size=(7,1)
        )
        self.branch0_4 = ConvolutionalBlock(
            filters=320,
            kernel_size=(3,3),
            strides=(2,2),
            padding='same'
        )

        self.branch1_1 = ConvolutionalBlock(
            filters=192,
            kernel_size=(1,1)
        )
        self.branch1_2 = ConvolutionalBlock(
            filters=192,
            kernel_size=(3,3),
            strides=(2,2),
            padding='same'
        )

        self.branch2 = tf.keras.layers.MaxPooling2D(
            pool_size=(3,3),
            strides=(2,2),
            padding='same'
        )

        self.output_layer = tf.keras.layers.Concatenate()

    def call(self, input_tensor):
        """ Perform forward pass."""
        x_branch0 = self.branch0_1(input_tensor)
        x_branch0 = self.branch0_2(x_branch0)
        x_branch0 = self.branch0_3(x_branch0)
        x_branch0 = self.branch0_4(x_branch0)

        x_branch1 = self.branch1_1(input_tensor)
        x_branch1 = self.branch1_2(x_branch1)

        x_branch2 = self.branch2(input_tensor)

        x_output = self.output_layer([ x_branch0, x_branch1, x_branch2 ])

        return x_output

class InceptionStemBlock(tf.keras.Model):
    """ Initial layers before """
    def __init__(self):
        """ Initialize layers."""
        super(InceptionStemBlock, self).__init__(name='')

        self.stem_conv1 = ConvolutionalBlock(
            filters=32,
            kernel_size=(3,3),
            strides=(2,2),
            padding='same'
        )
        self.stem_conv2 = ConvolutionalBlock(
            filters=32,
            kernel_size=(3,3),
            # strides=(1,1),
            padding='same'
        )
        self.stem_conv3 = ConvolutionalBlock(
            filters=64,
            kernel_size=(3,3)
        )

        self.stem_branch1a = tf.keras.layers.MaxPooling2D(
            pool_size=(3,3),
            strides=(2,2),
            padding='same'
        )
        self.stem_branch1b = ConvolutionalBlock(
            filters=96,
            kernel_size=(3,3),
            strides=(1,1),
            padding='same'
        )
        self.stem_concat1 = tf.keras.layers.Concatenate()

        self.stem_branch2a_1 = ConvolutionalBlock(
            filters=64,
            kernel_size=(1,1)
            # strides=(1,1)
        )
        self.stem_branch2a_2 = ConvolutionalBlock(
            filters=96,
            kernel_size=(3,3),
            padding='same'
            # strides=(1,1)
        )
        self.stem_branch2b_1 = ConvolutionalBlock(
            filters=64,
            kernel_size=(1,1)
            # strides=(1,1)
        )
        self.stem_branch2b_2 = ConvolutionalBlock(
            filters=64,
            kernel_size=(7,1)
            # strides=(1,1)
        )
        self.stem_branch2b_3 = ConvolutionalBlock(
            filters=64,
            kernel_size=(1,7)
            # strides=(1,1)
        )
        self.stem_branch2b_4 = ConvolutionalBlock(
            filters=96,
            kernel_size=(3,3),
            padding='same'
            # strides=(1,1)
        )
        self.stem_concat2 = tf.keras.layers.Concatenate()

        self.stem_branch3a = ConvolutionalBlock(
            filters=192,
            kernel_size=(3,3),
            padding='same'
            # strides=(1,1)
        )
        self.stem_branch3b = tf.keras.layers.MaxPooling2D(
            pool_size=(2,2),
            strides=None,
            padding='same'
        )

        self.output_layer = tf.keras.layers.Concatenate()

    def call(self, input_tensor):
        """ Perform forward pass."""
        x_conv1 = self.stem_conv1(input_tensor)
        x_conv2 = self.stem_conv2(x_conv1)
        x_conv3 = self.stem_conv3(x_conv2)

        x_branch1a = self.stem_branch1a(x_conv3)
        x_branch1b = self.stem_branch1b(x_conv2)

        x_branch1 = self.stem_concat1([x_branch1a, x_branch1b])

        x_branch2a = self.stem_branch2a_1(x_branch1)
        x_branch2a = self.stem_branch2a_2(x_branch2a)

        x_branch2b = self.stem_branch2b_1(x_branch1)
        x_branch2b = self.stem_branch2b_2(x_branch2b)
        x_branch2b = self.stem_branch2b_3(x_branch2b)
        x_branch2b = self.stem_branch2b_4(x_branch2b)

        x_branch2 = self.stem_concat2([x_branch2a, x_branch2b])

        x_branch3a = self.stem_branch3a(x_branch2)
        x_branch3b = self.stem_branch3b(x_branch2)

        x_output = self.output_layer([x_branch3a, x_branch3b])

        return x_output
