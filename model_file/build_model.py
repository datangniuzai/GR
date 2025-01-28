import tensorflow as tf

import config as cf
from model_file.GAT_GRU import GatGru


def creat_model():

    _, count_channel, _ = cf.feature_shape

    gat_gru_layer = GatGru(
        in_channels= 5,
        out_channels= 48,
        attn_heads= [6,1],
        hid_units= [8],
        dropout_rate_in=0.2,
        dropout_rate_out=0.3
    )

    layer_input_adjacency = tf.keras.Input(shape=(count_channel,count_channel), dtype=tf.float32,name='layer_input_adjacency')

    layer_input_graph = tf.keras.Input(shape=cf.feature_shape, name='layer_input_graph')

    layer_gat_gru= gat_gru_layer(layer_input_adjacency,layer_input_graph,None,last_layer=True)

    layer_flatten = tf.keras.layers.Flatten(name='layer_flatten')(layer_gat_gru)

    layer_dense = tf.keras.layers.Dense(units=128, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.001), name='layer_dense')(layer_flatten)

    layer_output = tf.keras.layers.Dense(units=cf.gesture_num, activation='softmax', name='output_layer_last')(layer_dense)

    model = tf.keras.Model(inputs=[layer_input_adjacency,layer_input_graph], outputs=layer_output)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=880,
    decay_rate=0.45,
    staircase=False
    )
   
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    return model
  
