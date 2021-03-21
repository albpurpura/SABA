import logging
import os
import numpy as np

np.random.seed(0)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
import tensorflow.compat.v1 as tf

tf.get_logger().setLevel('ERROR')
tf.logging.set_verbosity(tf.logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}
tf.get_logger().setLevel(logging.ERROR)
tf.set_random_seed(0)
tf.disable_v2_behavior()


class AutoEncoder:
    def __init__(self, femb_size, hidd_size_enc, hidd_size_sa, seq_size, learning_rate, seed=42):
        self.global_step = tf.Variable(0, trainable=False)
        self.training = tf.placeholder(tf.bool, None)
        self.masked_data = tf.placeholder(tf.float32, (None, seq_size, seq_size + 1), name='masked_data')
        self.true_data = tf.placeholder(tf.float32, (None, seq_size), name='true_data')
        self.masked_indices_mask = tf.placeholder(tf.float32, (None, seq_size), name='masked_indices_mask')
        self.feature_emb_matrix = tf.get_variable(shape=(seq_size + 1, femb_size),
                                                  initializer=tf.initializers.glorot_normal(seed=seed), trainable=True,
                                                  name='feature_emb_matrix')
        embedded_seq = tf.einsum('pe, bsp -> bse', self.feature_emb_matrix, self.masked_data)

        output_layer = tf.keras.layers.Dense(1)

        if hidd_size_sa > 0:
            sal = SelfAttentionLayer(femb_size, hidd_size_sa, seed)
            saw, hidd_r = sal.call(embedded_seq, self.training)
        else:
            hidd_l = tf.keras.layers.Dense(hidd_size_enc)
            hidd_r = hidd_l(embedded_seq)

        self.reconstructed_seq = tf.squeeze(output_layer(hidd_r), axis=-1)
        self.reconstructed_seq = tf.sigmoid(self.reconstructed_seq)
        self.loss = tf.reduce_sum(
            tf.multiply(tf.abs(self.true_data - self.reconstructed_seq), self.masked_indices_mask)) / tf.reduce_sum(
            self.masked_indices_mask)

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = optimizer.minimize(self.loss, global_step=self.global_step)
        self.init_op = tf.group(tf.compat.v1.global_variables_initializer(),
                                tf.compat.v1.local_variables_initializer())
        self.train_op = tf.group([train_op, update_ops])
        self.saver = tf.train.Saver(max_to_keep=None)


class SelfAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, input_data_size, proj_space_size, seed):
        super(SelfAttentionLayer, self).__init__()
        self.proj_space_size = proj_space_size
        self.k = tf.get_variable(name='K', shape=(input_data_size, self.proj_space_size),
                                 regularizer=tf.keras.regularizers.l2(),
                                 dtype=tf.float32, initializer=tf.initializers.glorot_normal(seed=seed), trainable=True)
        self.q = tf.get_variable(name='Q', shape=(input_data_size, self.proj_space_size),
                                 regularizer=tf.keras.regularizers.l2(),
                                 dtype=tf.float32, initializer=tf.initializers.glorot_normal(seed=seed), trainable=True)
        self.v = tf.get_variable(name='V', shape=(input_data_size, self.proj_space_size),
                                 regularizer=tf.keras.regularizers.l2(),
                                 dtype=tf.float32, initializer=tf.initializers.glorot_normal(seed=seed), trainable=True)

    def call(self, embdedded_features_vectors, training):
        Q = tf.einsum('eo, bse->bso', self.q, embdedded_features_vectors)
        K = tf.einsum('eo, bse->bso', self.k, embdedded_features_vectors)
        V = tf.einsum('eo, bse->bso', self.v, embdedded_features_vectors)

        QK = tf.matmul(Q, K, transpose_b=True)
        QK = QK / tf.sqrt(tf.cast(self.proj_space_size, tf.float32))
        interaction_weights = tf.reduce_sum(QK, axis=-1)
        att_w = tf.nn.softmax(interaction_weights, axis=-1)
        output = tf.layers.dropout(tf.einsum('bso,bs->bso', V, att_w), rate=0.5, training=training)
        output = tf.nn.l2_normalize(output)
        return att_w, output
