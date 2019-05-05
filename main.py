import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from lib.gcn import GraphConv, TemporalConv
from lib.graph import Graph

skeleton = Graph("sbu", "spatial")

input_features = Input([skeleton.num_node, skeleton.num_node, 3])
input_A = Input([skeleton.A.shape])
x, A = GraphConv(3, t_kernels=3)([input_features, input_A])
x = TemporalConv(A, filters=-1, t_kernels=1, t_strides=1,
                 in_batchnorm=True, out_batchnorm=True, dropout=0,)
