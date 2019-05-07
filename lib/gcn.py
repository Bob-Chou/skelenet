import numpy as np
import tensorflow as tf


def check_adjacent(A):
    """Check whether adjacent matrix A is valid and convert it to valid format

    :param A: adjacent matrix
    :return: formatted A
    """
    if not isinstance(A, tf.Tensor):
        raise ValueError("A must be a tf.Tensor but find {}".format(type(A)))
    if len(A.shape) == 3:
        try:
            assert A.shape[1] == A.shape[2]
        except AssertionError:
            raise ValueError("Input adjacent matrix must have the same"
                             "shape at last two dimension, but find {} "
                             "and {}".format(A.shape[1], A.shape[2]))
    elif len(A.shape) == 2:
        try:
            assert A.shape[0] == A.shape[1]
        except AssertionError:
            raise ValueError("Input adjacent matrix must have the same"
                             "shape at last two dimension, but find {} "
                             "and {}".format(A.shape[0], A.shape[1]))
        else:
            A = tf.expand_dims(A, 0)
    else:
        raise ValueError("Input adjacent matrix shape must in dimension 2"
                         "or 3 but find {}".format(len(A.shape)))
    return A


class GraphConv(tf.keras.layers.Layer):
    """Basic graphic convolutional layer for keras"""
    def __init__(self, filters, t_kernels=1, t_strides=1, **kwargs):
        super(GraphConv, self).__init__()
        self.filters = filters
        self.t_kernels = t_kernels // 2 * 2 + 1
        kwargs["padding"] = "same"
        kwargs["strides"] = (t_strides, 1)
        self.kwargs = kwargs
        self.Conv2D = None
        self.Reshape = None

    def build(self, input_shape):
        try:
            x_shape, A_shape = input_shape
        except ValueError:
            raise ValueError("Expected input_shape are [x.shape, A.shape]!")

        if len(x_shape) != 4:
            raise ValueError("Input x shape should be 3 dimension in "
                             "[timesteps, nodes, channels], but find "
                             "{} of {}".format(len(x_shape), x_shape[1:]))
        if len(A_shape) != 3:
            raise ValueError("Input A shape should be 3 dimension in "
                             "[label_number, nodes, nodes], but find "
                             "{} of {}".format(len(A_shape), A_shape))
        _, T, N, C = x_shape

        self.Conv2D = tf.keras.layers.Conv2D(self.filters * A_shape[0],
                                             (self.t_kernels, 1),
                                             **self.kwargs)
        self.Reshape = tf.keras.layers.Reshape((-1, T, N, A_shape[0],
                                                C//A_shape[0]))
        super(GraphConv, self).build(input_shape)

    def call(self, inputs, **kwargs):
        try:
            x, A = inputs
        except ValueError:
            raise ValueError("Expected inputs are [x, A]!")
        A = check_adjacent(A)
        x = self.Conv2D(x)
        x = self.Reshape(x)
        # compute the graphic conv:
        # f_out = sum_j{A_j * (X * W_j)}
        x = tf.einsum("btnkc,knm->btmc", x, A)
        return x, A

    def compute_output_shape(self, input_shape):
        x_shape, A_shape = input_shape
        return [np.asarray(
            self.Conv2D.compute_output_shape(x_shape).tolist()[:-1] +
            [self.filters]), A_shape]


class TemporalConv(tf.keras.layers.Layer):
    """Basic graphic convolutional layer for keras"""
    def __init__(self, filters=-1, t_kernels=3, t_strides=1,
                 in_batchnorm=True, out_batchnorm=True, dropout=0, **kwargs):
        super(TemporalConv, self).__init__()
        self.filters = filters
        self.t_kernels = t_kernels // 2 * 2 + 1
        kwargs["padding"] = "same"
        kwargs["strides"] = (t_strides, 1)
        self.kwargs = kwargs

        if in_batchnorm:
            self.BatchNorm0 = tf.keras.layers.BatchNormalization(axis=-1)
        else:
            self.BatchNorm0 = tf.keras.layers.Lambda(lambda x: x)
        if out_batchnorm:
            self.BatchNorm1 = tf.keras.layers.BatchNormalization(axis=-1)
        else:
            self.BatchNorm1 = tf.keras.layers.Lambda(lambda x: x)
        self.Dropout = tf.keras.layers.Dropout(dropout)
        self.Conv2D = None

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("Input shape should be 3 dimension in "
                             "[timesteps, nodes, channels], but find "
                             "{} of {}".format(len(input_shape), input_shape))
        if self.filters < 0:
            self.filters = input_shape[-1]
        self.Conv2D = tf.keras.layers.Conv2D(self.filters,
                                             (self.t_kernels, 1),
                                             activation=tf.keras.layers.ReLU,
                                             **self.kwargs)
        super(TemporalConv, self).build(input_shape)

    def call(self, inputs, **kwargs):
        inputs = self.BatchNorm0(inputs)
        inputs = self.Conv2D(inputs, **kwargs)
        inputs = self.BatchNorm1(inputs)
        inputs = self.Dropout(inputs)
        return inputs

    def compute_output_shape(self, input_shape):
        return self.Dropout.compute_output_shape(input_shape)
