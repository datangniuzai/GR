import tensorflow as tf
from model_file.GAT_GRU import GATGRU
import config as cf
def creat_model():
    adjacency = tf.keras.Input(shape=(64,64), dtype=tf.float32,name='input_1')
    features_cnn = tf.keras.Input(shape=cf.feature_shape, name='input_2')
    gclstm_1 = GATGRU(6,48,[6,1],[8])
    gcn_1= gclstm_1(adjacency,features_cnn,None,last_layer=True)
    flatten_1 = tf.keras.layers.Flatten(name='flatten_1')(gcn_1)
    dense1 = tf.keras.layers.Dense(units=128, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.001), name='dense_last')(flatten_1)
    output_layer = tf.keras.layers.Dense(units=cf.gesture_num, activation='softmax', name='output_layer_last')(dense1)
    model = tf.keras.Model(inputs=[adjacency,features_cnn], outputs=output_layer)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=880,
    decay_rate=0.45,
    staircase=False)
   
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model
  
