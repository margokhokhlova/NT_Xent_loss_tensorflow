import random
import tensorflow as tf
import numpy as np

@tf.function
def contrastive_loss(xi, xj,  tau=1, normalize=False):
        ''' this loss was inspred by the torch implementation here: https://github.com/mdiephuis/SimCLR/
        the inputs:
        xi, xj: image features extracted from a batch of images 2N, composed of N matching paints
        tau: temperature parameter
        normalize: normalize or not. seem to not be very useful, so better to try without.
        '''

        x = tf.keras.backend.concatenate((xi, xj), axis=0)

        sim_mat = tf.keras.backend.dot(x, tf.keras.backend.transpose(x))

        if normalize:
            sim_mat_denom = tf.keras.backend.dot(tf.keras.backend.l2_normalize(x, axis=1).unsqueeze(1), tf.keras.backend.l2_normalize(x, axis=1).unsqueeze(1).T)
            sim_mat = sim_mat / sim_mat_denom.clamp(min=1e-16)

        sim_mat = tf.keras.backend.exp(sim_mat /tau)

        if normalize:
            sim_mat_denom = tf.keras.backend.l2_normalize(xi, dim=1) * tf.keras.backend.l2_normalize(xj, axis=1)
            sim_match = tf.keras.backend.exp(tf.keras.backend.sum(xi * xj, axis=-1) / sim_mat_denom / tau)
        else:
            sim_match = tf.keras.backend.exp(tf.keras.backend.sum(xi * xj, axis=-1) / tau)

        sim_match = tf.keras.backend.concatenate((sim_match, sim_match), axis=0)

        norm_sum = tf.keras.backend.exp(tf.keras.backend.ones(tf.keras.backend.shape(x)[0]) / tau)

        return tf.math.reduce_mean(-tf.keras.backend.log(sim_match / (tf.keras.backend.sum(sim_mat, axis=-1) - norm_sum)), name='contrastive_loss')



class ContrastiveLossLayer(tf.keras.layers.Layer):

    def __init__(self, tau=1
                 , normalize = False, name=None):
        super(ContrastiveLossLayer, self).__init__(name=name)
        self._tau = tau
        self.normalize = normalize

    def __call__(self, xi, xj):
        return super(ContrastiveLossLayer, self).__call__([xi,xj])

    def call(self, inputs):
        loss = contrastive_loss(*inputs, tau=self._tau, normalize = self.normalize)
        self.add_loss(loss)
        return loss


#
# @tf.function
# def contrastive_loss(xi, xj,  tau=1, normalize=False):
#         x = tf.concat([xi, xj], axis=0)
#         sim_mat = tf.tensordot(x, tf.transpose(x), axes = 1)
#         if normalize:
#             sim_mat_denom = tf.keras.backend.dot(tf.keras.backend.norm(x, dim=1).unsqueeze(1), tf.keras.backend.norm.norm(x, dim=1).unsqueeze(1).T)
#             sim_mat = sim_mat / sim_mat_denom.clamp(min=1e-16)
#
#         sim_mat = tf.math.exp(sim_mat /tau)
#
#         # top
#         if normalize:
#             sim_mat_denom = tf.keras.backend.norm(xi, dim=1) * tf.keras.backend.norm(xj, axis=1)
#             sim_match = tf.keras.backend.exp(tf.keras.backend.sum(xi * xj, axis=-1) / sim_mat_denom / tau)
#         else:
#             sim_match = tf.math.exp(tf.keras.backend.sum(xi * xj, axis=-1) / tau)
#
#         sim_match = tf.concat([sim_match, sim_match], axis=0)
#
#         norm_sum = tf.math.exp(tf.ones(tf.shape(x)[0]) / tau)
#         return tf.reduce_mean(tf.cast(tf.math.reduce_mean(-tf.math.log(sim_match / (tf.reduce_sum(sim_mat, axis=-1) - norm_sum))), tf.dtypes.float32))

if __name__ == '__main__':
    random.seed(30)
    batch_size = 2
    feature_size = 16
    x =np.random.rand(12, 256)
    # x = np.asarray([[0.1, 0.1, 0.2, 0.3, 0.4],  # 1st row
    #                 [0.2, 0.6, 0.7, 0.8, 0.9],  # 2nd row
    #                 [0.3, 0.6, 0.7, 0.8, 0.9],  # 3rd row
    #                 [0.4, 0.6, 0.7, 0.8, 0.9],  # 4th row
    #                 [0.5, 0.6, 0.7, 0.8, 0.9]  # 5th row
    #                 ], dtype=np.float)
    print(x)
    x =tf.convert_to_tensor(x, dtype='float32')
    loss = contrastive_loss(x,x)
    print(tf.keras.backend.eval(loss))

