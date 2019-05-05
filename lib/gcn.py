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
        self.kernels = A.shape[0]
        self.in_shape = None
        t_kernels = t_kernels // 2 * 2 + 1
        kwargs["padding"] = "same"
        kwargs["strides"] = (t_strides, 1)
        self.Conv2D = tf.keras.layers.Conv2D(filters * self.kernels,
                                           (t_kernels, 1),
                                           **kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("Input shape should be 3 dimension in "
                             "[timesteps, nodes, channels], but find "
                             "{} of {}".format(len(input_shape), input_shape))
        self.Conv2D = tf.keras.layers.Conv2D(self.filters * self.kernels,
                                             (self.t_kernels, 1),
                                             activation=tf.keras.layers.ReLU,
                                             **self.kwargs)
        self.in_shape = input_shape
        self.conv.build(input_shape)
        super(GraphConv, self).build(input_shape)

    def call(self, inputs, **kwargs):
        try:
            x, A = inputs
        except ValueError:
            raise ValueError("Expected inputs are [x, A]!")
        A = check_adjacent(A)
        x = self.Conv2D(x, **kwargs)
        T, N, C = self.in_shape
        inputs = tf.reshape(x, (-1, T, N, self.kernels, C//self.kernels))
        # compute the graphic conv:
        # f_out = sum_j{A_j * (X * W_j)}
        x = tf.einsum("btnkc,knm->btmc", inputs, A)
        return x, A


class TemporalConv(tf.keras.layers.Layer):
    """Basic graphic convolutional layer for keras"""
    def __init__(self, filters=-1, t_kernels=1, t_strides=1,
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
        self.BatchNorm0.build(input_shape)
        self.Conv2D.build(input_shape)
        self.BatchNorm1.build(input_shape)
        self.Dropout.build(input_shape)
        super(TemporalConv, self).build(input_shape)

    def call(self, inputs, **kwargs):
        inputs = self.BatchNorm0(inputs)
        inputs = self.Conv2D(inputs, **kwargs)
        inputs = self.BatchNorm1(inputs)
        inputs = self.Dropout(inputs)
        return inputs
