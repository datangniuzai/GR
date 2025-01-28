import tensorflow as tf


class GAT(tf.keras.layers.Layer):
    def __init__(self,attn_heads, hid_units, activation=tf.nn.elu, residual=False,dropout_rate_in=0.2,dropout_rate_out=0.2):
        super(GAT, self).__init__()
        
        self.attn_heads = attn_heads
        self.hid_units = hid_units
        self.activation = activation
        self.residual = residual
        self.dropout_rate_in = dropout_rate_in
        self.dropout_rate_out = dropout_rate_out
        self.conv_layers1 = tf.keras.layers.Conv1D(filters=self.hid_units[0],kernel_size=1,name='layer1')
        self.conv_layers2 = tf.keras.layers.Conv1D(filters=1,kernel_size=1,name='layer2')
        self.dropout_in = tf.keras.layers.Dropout(rate=self.dropout_rate_in)
        self.dropout_out = tf.keras.layers.Dropout(rate=self.dropout_rate_out)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        
    def attn_head(self,input,bias_mat):
        
       
        seq_fts = self.conv_layers1(input)
        f_1 = self.conv_layers2(seq_fts)
        f_2 = self.conv_layers2(seq_fts)
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)
        coefs = self.dropout_out(coefs)
        seq_fts = self.dropout_in(seq_fts)
        vals = tf.matmul(coefs, seq_fts)
        return self.activation(vals)

    def call(self, inputs,bias_mat):
        attns = []
        for _ in range(self.attn_heads[0]):
            out_1=self.attn_head(input=inputs,bias_mat=bias_mat)
            attns.append(out_1)
        h_1 = tf.concat(attns, axis=-1) 
        h_1 = self.batch_norm(h_1)
        return h_1
    def get_config(self):

        config = super().get_config()

        config.update({
            "attn_heads": self.attn_heads,
            "hid_units": self.hid_units,
            "activation": self.activation,
            "residual": self.residual,
            "dropout_rate_in": self.dropout_rate_in,
            "dropout_rate_out":self.dropout_rate_out
        })

        return config

        