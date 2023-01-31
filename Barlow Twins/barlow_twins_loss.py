import tensorflow as tf

class Custom_barlow_twins(tf.keras.losses.Loss):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def normalize(self, output: tf.Tensor) -> tf.Tensor:
        return (output - tf.reduce_mean(output, axis=0)) / tf.math.reduce_std(output, axis=0)

    def cross_corr_matrix(self, z_a_norm: tf.Tensor, z_b_norm: tf.Tensor) -> tf.Tensor:
        return (tf.transpose(z_b_norm) @z_a_norm) / self.batch_size

    def call(self, z_a, z_b):        
        #z_a_norm, z_b_norm = self.normalize(z_a), self.normalize(z_b)
        cross_matrix = self.cross_corr_matrix(z_a, z_b)
        loss_1 = cross_matrix - tf.eye(cross_matrix.shape[0], cross_matrix.shape[1])
        loss_2 = tf.reduce_sum(loss_1**2)
        return loss_2