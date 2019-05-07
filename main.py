import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from lib.gcn import GraphConv, TemporalConv
from lib.graph import Graph

skeleton = Graph("sbu", "spatial")

input_features = Input([30, skeleton.num_node, 3], dtype="float32")
x = tf.keras.layers.Conv2D(64 * 3, (3, 1), padding="same")(input_features)
input_A = Input(tensor=tf.keras.backend.constant(skeleton.A))
x, A = GraphConv(64, t_kernels=3)([input_features, input_A])
x = TemporalConv(64, dropout=0.5)(x)
x, A = GraphConv(128, t_kernels=3)([x, A])
x = TemporalConv(128, dropout=0.5)(x)
print(x.shape)