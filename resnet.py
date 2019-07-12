import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, ReLU, Lambda, Add, BatchNormalization, Multiply


def _glorot_initializer_conv2d(x, prev_units, num_units, mapsize, stddev_factor=1.0):
    """Initialization in the style of Glorot 2010.

    stddev_factor should be 1.0 for linear activations, and 2.0 for ReLUs"""

    stddev = np.sqrt(stddev_factor / (np.sqrt(prev_units * num_units) * mapsize * mapsize))
    initw = keras.initializers.TruncatedNormal(mean=0.0, stddev=stddev)
    return initw


def add_conv2d(x, num_units, mapsize=1, stride=1, stddev_factor=1.0):
    """Adds a 2D convolutional layer."""

    assert len(x.shape) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels)"

    prev_units = x.get_shape()[-1]

    # Weight term and convolution
    initw = _glorot_initializer_conv2d(x, prev_units, num_units,
                                       mapsize, stddev_factor=stddev_factor)

    out = Conv2D(num_units, mapsize, strides=stride, padding='same', use_bias=True,
                 kernel_initializer=initw, bias_initializer='zero')(x)

    return out


def add_conv2d_transpose(x, num_units, mapsize=1, stride=1, stddev_factor=1.0):
    """Adds a transposed 2D convolutional layer"""

    assert len(x.shape) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels)"

    prev_units = x.get_shape()[-1]

    # Weight term and convolution
    initw = _glorot_initializer_conv2d(x, prev_units, num_units,
                                       mapsize, stddev_factor=stddev_factor)

    out = Conv2DTranspose(num_units, mapsize, strides=stride, padding='same', use_bias=True,
                          kernel_initializer=initw, bias_initializer='zero')(x)

    return out


def add_residual_block(inputs, num_units, mapsize=3, num_layers=2, stddev_factor=1e-3):
    """Adds a residual block as per Arxiv 1512.03385, Figure 3"""

    # Add projection in series if needed prior to shortcut
    if num_units != int(inputs.get_shape()[-1]):
        inputs = add_conv2d(inputs, num_units, mapsize=1, stride=1, stddev_factor=1.)

    x = tf.identity(inputs)

    # Residual block
    for _ in range(num_layers):
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = add_conv2d(x, num_units, mapsize=mapsize, stride=1, stddev_factor=stddev_factor)

    outputs = Add()([inputs, x])
    return outputs


def Fourier(x):
    x_complex = tf.complex(x[..., 0], x[..., 1])
    y_complex = tf.signal.fft2d(x_complex)
    return y_complex


def expand_dims(x):
    return tf.expand_dims(x, -1)


def add_dc_layer(x, features, mask):
    # add dc connection for each block
    # parameters
    mix_DC = 1  # 0.95

    mask = mask*mix_DC
    first_layer = tf.identity(features)
    feature_kspace = Lambda(Fourier)(first_layer)
    projected_kspace = Multiply()([feature_kspace, mask])

    # get output and input
    last_layer = tf.identity(x)
    gene_kspace = Lambda(Fourier)(last_layer)
    gene_kspace = Multiply()([gene_kspace, (1.0 - mask)])

    corrected_kspace = Add()([projected_kspace, gene_kspace])

    # inverse fft
    corrected_complex = tf.signal.ifft2d(corrected_kspace)
    corrected_mag = tf.abs(corrected_complex)
    corrected_mag = Lambda(expand_dims, name='gene_output')(corrected_mag)

    return corrected_mag


def resnet_o(img_width=64, img_height=64, channels=2):
    inputs = Input(shape=[img_width, img_height, channels], name='zero_fill')
    mask = Input(shape=[img_width, img_height], dtype=tf.complex64, name='under_mask')

    res_units = [128, 128, 128, 128, 128]
    mapsize = 3
    x = tf.identity(inputs)

    for ru in range(len(res_units) - 1):
        nunits = res_units[ru]

        for j in range(2):
            x = add_residual_block(x, nunits, mapsize=mapsize)

        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = add_conv2d_transpose(x, nunits, mapsize=mapsize, stride=1, stddev_factor=1.)

    nunits = res_units[-1]
    x = add_conv2d(x, nunits, mapsize=mapsize, stride=1, stddev_factor=2.)
    x = ReLU()(x)

    x = add_conv2d(x, nunits, mapsize=1, stride=1, stddev_factor=2.)
    x = ReLU()(x)

    # Last layer is sigmoid with no batch normalization
    x = add_conv2d(x, channels, mapsize=1, stride=1, stddev_factor=2.)

    # Add dc layer
    gene_output = add_dc_layer(x, inputs, mask)

    return keras.Model(inputs=[inputs, mask], outputs=gene_output, name='resnet')
