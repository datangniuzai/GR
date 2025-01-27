import tensorflow as tf

from model_file.GAT import GAT


class GatGru(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, attn_heads, hid_units,
                 dropout_rate_in, dropout_rate_out, **kwargs):
        super(GatGru, self).__init__(**kwargs)


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attn_heads = attn_heads
        self.hid_units = hid_units
        self.dropout_rate_in = dropout_rate_in
        self.dropout_rate_out = dropout_rate_out

        self._create_parameters_and_layers()

        self.W_z = None
        self.Z_bias = None
        self.W_h = None
        self.H_bias = None
        self.layer_dropout_out = None
        self.layer_dropout_in = None
        self.batch_norm = None

    def build(self, input_shape):
        """Initialize weights and biases used in the layer."""
        self.W_z = self.add_weight(
            shape=(self.in_channels, self.out_channels),
            initializer='he_normal',
            trainable=True,
            name='W_z'
        )
        
        self.Z_bias = self.add_weight(
            shape=(1, self.out_channels),
            initializer='he_normal',
            trainable=True,
            name='Z_bias'
        )

        self.W_h = self.add_weight(
            shape=(self.in_channels, self.out_channels),
            initializer='he_normal',
            trainable=True,
            name='W_h'
        )

        self.H_bias = self.add_weight(
            shape=(1, self.out_channels),
            initializer='he_normal',
            trainable=True,
            name='H_bias'
        )

        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.layer_dropout_in = tf.keras.layers.Dropout(rate=self.dropout_rate_in)
        self.layer_dropout_out = tf.keras.layers.Dropout(rate=self.dropout_rate_out)

    def _calculate_update_gate_parameters_and_layers(self):
        return GAT(self.attn_heads, self.hid_units)

    def _calculate_hidden_gate_parameters_and_layers(self):
        return GAT(self.attn_heads, self.hid_units)

    def _create_parameters_and_layers(self):

        self.conv_z= self._calculate_update_gate_parameters_and_layers()
        self.conv_h= self._calculate_hidden_gate_parameters_and_layers()

    def _set_hidden_state(self, x, h):

        if h is None:

            batch_size = tf.shape(x)[0]
            shape = (batch_size, x.shape[2], self.out_channels)
            h = tf.zeros(shape, dtype=tf.float32)

        return h

    def _calculate_update_gate(self, x, edge_index, h):
        z = self.conv_z(inputs=x, bias_mat=edge_index)
        z = z +tf.matmul(x,self.W_z)+self.Z_bias+h
        z = tf.sigmoid(z)
        return z

    def _calculate_hidden_state(self, x, edge_index, h, z):
        t = self.conv_h(inputs=x, bias_mat=edge_index)
        t = t + h +tf.matmul(x,self.W_h)+self.H_bias
        t = tf.tanh(t)
        h = z * h +(1-z)*t
        return h

    def call(self,edge_index, x, h=None, last_layer=True):

        h = self._set_hidden_state(x, h)
        out_puts = []

        for i in range(x.shape[1]):
            z = self._calculate_update_gate(x[:, i, :, :], edge_index, h)
            h = self._calculate_hidden_state(x[:, i, :, :],edge_index, h, z)
            h = self.layer_dropout_out(h)
            outdata = h
            outdata = tf.expand_dims(outdata, axis=-1)
            out_puts.append(outdata)

        out_puts = tf.concat(out_puts, axis=-1)
        if last_layer:
            out_puts = h
            h = self.batch_norm(out_puts)
        else:
            out_puts = tf.transpose(out_puts, [0, 3, 1, 2])
            h = self.batch_norm(out_puts)
        h = self.layer_dropout_out(h)
        return h

    def get_config(self):

        config = super().get_config()

        config.update({
            "in_channels":self.in_channels,
            "out_channels":self.out_channels,
            "attn_heads": self.attn_heads,
            "hid_units": self.hid_units,
            "dropout_rate_in": self.dropout_rate_in,
            "dropout_rate_out":self.dropout_rate_out
        })
        return config

