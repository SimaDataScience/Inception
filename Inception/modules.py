""" Inception blocks."""
import tensorflow as tf

class ConvolutionalBlock(tf.keras.Model):
    """ 2D convolutional layer with batch normalization and relu activaiton."""
    def __init__(self, **kwargs):
        """ Initialize layers."""
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
    def __init__(self, filters, reductions):
        """ Initialize layers.

        Args:
            filters (list): List of number of filters for layers.
            reductions (list): List of the number of filters for reductions.
        """
        super(InceptionBlockA, self).__init__(name='')
        filter_1, filter_3, filter_5 = filters
        reduction_3_1, reduction_3_2, reduction_maxpool = reductions

        self.conv_3a1 = ConvolutionalBlock(
            filters=reduction_3_1,
            kernel_size=(1,1),
            padding='same',
            activation='relu'
        )
        self.conv_3a2 = ConvolutionalBlock(
            filters=filter_3,
            kernel_size=(3,3),
            padding='same',
            activation='relu'
        )

        # Two 3x3 convolutions after reduction, instead of 5x5 layer from original paper.
        self.conv_3b1 = ConvolutionalBlock(
            filters=reduction_3_2,
            kernel_size=(1,1),
            padding='same',
            activation='relu'
        )
        self.conv_3b2 = ConvolutionalBlock(
            filters=reduction_3_2,
            kernel_sixe=(3,3),
            padding='same',
            activation='relu'
        )
        self.conv_3b3 = ConvolutionalBlock(
            filters=filter_5,
            kernel_sixe=(3,3),
            padding='same',
            activation='relu'
        )

        self.conv_maxpool_a = tf.keras.layers.MaxPooling2D(
            pool_size=(3,3)
        )
        self.conv_maxpool_b = ConvolutionalBlock(
            filters=reduction_maxpool,
            kernel_size=(1,1),
            padding='same',
            activation='relu'
        )

        self.conv_1 = ConvolutionalBlock(
            filters=filter_1,
            kernel_size=(1,1),
            padding='same',
            activation='relu'
        )

        self.output_layer = tf.keras.layers.concatenate()

    def call(self, input_tensor):
        """ Perform forward pass."""
        x_1 = self.conv_1(input_tensor)

        x_3a = self.conv_3a1(input_tensor)
        x_3a = self.conv_3a2(x_3a)

        x_3b = self.conv_3b1(input_tensor)
        x_3b = self.conv_3b2(x_3b)
        x_3b = self.conv_3b3(x_3b)

        x_maxpool = self.conv_maxpool_a(input_tensor)
        x_maxpool = self.conv_maxpool_b(x_maxpool)

        x_output = self.output_layer([ x_1, x_3a, x_3b, x_maxpool ])

        return x_output

class InceptionBlockB(tf.keras.Model):
    """ Computationally efficient factorization of nxn convolutional layer."""
    def __init__(self, filters, reductions, kernel_size):
        """ Initialize layers.

        Args:
            filters (list): List of number of filters for layers.
            reductions (list): List of the number of filters for reductions.
        """
        super(InceptionBlockB, self).__init__(name='')
        filter_1, filter_split_1, filter_split_n = filters
        reduction_split_1, reduction_split_n, reduction_maxpool = reductions

        self.conv_n_long1 = ConvolutionalBlock(
            filters=reduction_split_n,
            kernel_size=(1,1)
        )
        self.conv_n_long2 = ConvolutionalBlock(
            filters=filter_split_n,
            kernel_size=(1,kernel_size)
        )
        self.conv_n_long3 = ConvolutionalBlock(
            filters=filter_split_n,
            kernel_size=(kernel_size,1)
        )
        self.conv_n_long4 = ConvolutionalBlock(
            filters=filter_split_n,
            kernel_size=(1,kernel_size)
        )
        self.conv_n_long5 = ConvolutionalBlock(
            filters=filter_split_n,
            kernel_size=(kernel_size,1)
        )

        self.conv_n_short1 = ConvolutionalBlock(
            filters=reduction_split_1,
            kernel_size=(1,1)
        )
        self.conv_n_short2 = ConvolutionalBlock(
            filters=filter_split_1,
            kernel_size=(1,kernel_size)
        )
        self.conv_n_short3 = ConvolutionalBlock(
            filters=filter_split_1,
            kernel_size=(kernel_size,1)
        )

        self.conv_maxpool_a = tf.keras.layers.MaxPooling2D(
            pool_size=(3,3)
        )
        self.conv_maxpool_b = ConvolutionalBlock(
            filters=reduction_maxpool,
            kernel_size=(1,1),
            activation='relu'
        )

        self.conv_1 = ConvolutionalBlock(
            filters=filter_1,
            kernel_size=(1,1),
            activation='relu'
        )

        self.output_layer = tf.keras.layers.concatenate()

    def call(self, input_tensor):
        """ Perform forward pass."""
        x_long = self.conv_n_long1(input_tensor)
        x_long = self.conv_n_long2(x_long)
        x_long = self.conv_n_long3(x_long)
        x_long = self.conv_n_long4(x_long)
        x_long = self.conv_n_long5(x_long)

        x_short = self.conv_n_short1(input_tensor)
        x_short = self.conv_n_short2(x_short)
        x_short = self.conv_n_short3(x_short)

        x_maxpool = self.conv_maxpool_a(input_tensor)
        x_maxpool = self.conv_maxpool_b(x_maxpool)

        x_1 = self.conv_1(input_tensor)

        x_output = self.output_layer([x_long, x_short, x_maxpool, x_1])

        return x_output

class InceptionBlockC(tf.keras.Model):
    """ Inception module with expanded filter bank."""
    def __init__(self, filters: list, reductions: list):
        """ Initialize layers.

        Args:
            filters (list): List of number of filters for layers.
            reductions (list): List of the number of filters for reductions.
        """
        super(InceptionBlockC, self).__init__(name='')
        filter_1, filter_split_1, filter_split_3a, filter_split_3b = filters
        reduction_split_1, reduction_split_3, reduction_maxpool = reductions

        self.conv_split_3a = ConvolutionalBlock(
            filters=reduction_split_3,
            kernel_size=(1,1),
            padding='same',
            activation='relu'
        )
        self.conv_split_3b = ConvolutionalBlock(
            filters=filter_split_3a,
            kernel_size=(3,3),
            padding='same',
            activation='relu'
        )
        self.conv_split_3c = ConvolutionalBlock(
            filters=filter_split_3b,
            kernel_size=(1,3),
            padding='same',
            activation='relu'
        )
        self.conv_split_3d = ConvolutionalBlock(
            filters=filter_split_3b,
            kernel_size=(3,1),
            padding='same',
            activation='relu'
        )

        self.conv_split_1a = ConvolutionalBlock(
            filters=reduction_split_1,
            kernel_size=(1,1),
            padding='same',
            activation='relu'
        )
        self.conv_split_1b = ConvolutionalBlock(
            filters=filter_split_1,
            kernel_size=(1,3),
            padding='same',
            activation='relu'
        )
        self.conv_split_1c = ConvolutionalBlock(
            filters=filter_split_1,
            kernel_size=(3,1),
            padding='same',
            activation='relu'
        )

        self.conv_maxpool_a = tf.keras.layers.MaxPooling2D(
            pool_size=(3,3)
        )
        self.conv_maxpool_b = ConvolutionalBlock(
            filters=reduction_maxpool,
            kernel_size=(1,1),
            padding='same',
            activation='relu'
        )

        self.conv_1 = ConvolutionalBlock(
            filters=filter_1,
            kernel_size=(1,1),
            padding='same',
            activation='relu'
        )

        self.output_layer = tf.keras.layers.concatenate()

    def call(self, input_tensor):
        """ Perform forward pass."""
        x_split3 = self.conv_split_3a(input_tensor)
        x_split3 = self.conv_split_3b(x_split3)

        x_split3a = self.conv_split_3c(x_split3)
        x_split3b = self.conv_split_3d(x_split3)

        x_split1 = self.conv_split_1a(input_tensor)

        x_split1a = self.conv_split_1b(x_split1)
        x_split1b = self.conv_split_1c(x_split1)

        x_1 = self.conv_1(input_tensor)

        x_maxpool = self.conv_maxpool_a(input_tensor)
        x_maxpool = self.conv_maxpool_b(x_maxpool)

        x_output = self.output_layer(
            [
                x_split3a, x_split3b,
                x_split1a, x_split1b,
                x_maxpool,
                x_1
            ]
        )

        return x_output

class InceptionReductionA(tf.keras.Model):
    """ Dimension reduction block."""
    def __init__(self, filters, reductions):
        """ Initialize layers.

        Args:
            filters (list): List of number of filters for layers.
        """
        super(InceptionReductionA, self).__init__(name='')
        filter_deep, filter_shallow = filters
        reduction_deep, reduction_shallow = reductions

        self.conv_deep1 = ConvolutionalBlock(
            filters=reduction_deep,
            kernel_size=(1,1)
        )
        self.conv_deep2 = ConvolutionalBlock(
            filters=filter_deep,
            kernel_size=(3,3),
            strides=(1,1)
        )
        self.conv_deep3 = ConvolutionalBlock(
            filters=filter_deep,
            kernel_size=(3,3),
            strides=(2,2)
        )

        self.conv_shallow_1 = ConvolutionalBlock(
            filters=reduction_shallow,
            kernel_size=(1,1)
        )
        self.conv_shallow_2 = ConvolutionalBlock(
            filters=filter_shallow,
            kernel_size=(3,3),
            strides=(2,2)
        )

        self.conv_maxpool = tf.keras.layers.MaxPooling2D(
            pool_size=(3,3),
            strides=(2,2)
        )

        self.output_layer = tf.keras.layers.concatenate()

    def call(self, input_tensor):
        """ Perform forward pass."""
        x_deep = self.conv_deep1(input_tensor)
        x_deep = self.conv_deep2(x_deep)
        x_deep = self.conv_deep3(x_deep)

        x_shallow = self.conv_shallow_1(input_tensor)
        x_shallow = self.conv_shallow_2(input_tensor)

        x_maxpool = self.conv_maxpool(input_tensor)

        x_output = self.output_layer([x_deep, x_shallow, x_maxpool])

        return x_output

class InceptionReductionB(tf.keras.Model):
