import numpy as np
import tensorflow as tf
from contrastive_loss import ContrastiveLossLayer

# an example of a double input (siamese) network was found somewhere on forums dedicated to tensorflow
class L2Normalization(tf.keras.layers.Layer):

    def __init__(self, name=None):
        super(L2Normalization, self).__init__(name=name)

    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)


def create_base_model(input_shape=(28, 28)):
    input = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(input)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = L2Normalization()(x)
    return tf.keras.models.Model(input, x)


base_model = create_base_model()

input_a = tf.keras.layers.Input(shape=(28, 28), name="input_a")
input_b = tf.keras.layers.Input(shape=(28, 28), name="input_b")

output_a = base_model(input_a)
output_b = base_model(input_b)

# custom loss layer
outputs = ContrastiveLossLayer()(output_a, output_b)

model = tf.keras.models.Model([input_a, input_b], outputs=outputs)

model.compile(tf.keras.optimizers.Adam(1e-3))

model.summary()

sample_data = {
    'input_a': np.random.rand(1000, 28, 28),
    'input_b': np.random.rand(1000, 28, 28),
}

model.fit(sample_data, epochs=18)