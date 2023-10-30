from tensorflow.keras import regularizers
import tensorflow as tf

class RegularizedDenseLayer(tf.keras.layers.Layer):
    def __init__(self, units, l1=0.001, l2=0.001):
        super(RegularizedDenseLayer, self).__init__()
        self.units = units
        self.l1 = l1
        self.l2 = l2

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='he_uniform',
                                 trainable=True,
                                 regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2))

    def call(self, inputs):
        return tf.matmul(inputs, self.w)

