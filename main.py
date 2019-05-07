import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from lib.gcn import GraphConv, TempoConv, STConv
from lib.graph import Graph
skeleton = Graph("sbu", "spatial")

input_features = Input([30, skeleton.num_node, 3], dtype="float32")
input_A = Input(tensor=K.constant(skeleton.A))
x, A = STConv(64, kernels=(3, 3), strides=(1, 1))([input_features, input_A])
x, A = STConv(64, kernels=(3, 3), strides=(1, 1))([x, A])
x, A = STConv(64, kernels=(3, 3), strides=(1, 1))([x, A])
x, A = STConv(64, kernels=(3, 3), strides=(1, 1))([x, A])
x, A = STConv(128, kernels=(3, 3), strides=(2, 1))([x, A])
x, A = STConv(128, kernels=(3, 3), strides=(1, 1))([x, A])
x, A = STConv(128, kernels=(3, 3), strides=(1, 1))([x, A])
x, A = STConv(256, kernels=(3, 3), strides=(2, 1))([x, A])
x, A = STConv(256, kernels=(3, 3), strides=(1, 1))([x, A])
output_features, A = STConv(256, kernels=(3, 3), strides=(1, 1))([x, A])

model = Model(inputs=[input_features, input_A], outputs=output_features)
model.summary()