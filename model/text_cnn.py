import tensorflow as tf
import numpy as np
import sys
from sklearn.metrics import f1_score, accuracy_score

from .basic_model import BasicModel


class TextCNN(BasicModel):
    def __init__(self, matrix=None, maxlen=None, feature_keep_prob=0.6):
        super(TextCNN, self).__init__()

        self.matrix = matrix
        self.maxlen = maxlen
        self.feature_keep_prob = feature_keep_prob

        self.build_model()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def build_model(self):
        self.text_input = tf.placeholder(dtype=tf.int32, shape=[None, self.maxlen], name='text_input')
        self.label_input = tf.placeholder(dtype=tf.int32, shape=[None, ], name='label')

        self.feature_keep_prob_input = tf.placeholder(dtype=tf.float32, name='feature_keep_prob')

        with tf.variable_scope('embedding_layer'):
            embedding = tf.get_variable(
                name='embedding',
                shape=self.matrix.shape,
                initializer=tf.constant_initializer(self.matrix),
                trainable=False
            )
            text_embed = tf.nn.embedding_lookup(embedding, self.text_input)

        with tf.variable_scope('cnn_layer'):
            filter_sizes = [3, 5, 7, 9]
            filters = 64
            conv_result = []

            for filter_size in filter_sizes:
                conv = tf.layers.conv1d(text_embed, filters=64, kernel_size=filter_size,
                                        kernel_initializer=tf.truncated_normal_initializer,
                                        bias_initializer=tf.zeros_initializer, activation=tf.nn.relu)
                max_pool = tf.reduce_max(conv, axis=1)
                conv_result.append(max_pool)

        with tf.variable_scope('output_layer'):
            feature = tf.concat(conv_result, axis=-1)
            feature = tf.nn.dropout(feature, keep_prob=self.feature_keep_prob_input)
            out_ = tf.layers.dense(feature, units=1, activation=tf.nn.sigmoid)

            labels = tf.cast(self.label_input, tf.float32)
            labels = tf.expand_dims(labels, axis=-1)
            self.probs = out_
            self.loss = tf.losses.log_loss(labels=labels, predictions=out_)

        with tf.variable_scope('train_op'):
            self.train_op = tf.train.AdamOptimizer(0.01).minimize(self.loss)

    def next_batch(self, x, y=None, batch_size=64, shuffle=False):
        if shuffle:
            from sklearn import utils
            x, y = utils.shuffle(x, y)
        for idx in range(0, len(x), batch_size):
            s_idx = idx
            e_idx = idx + batch_size
            e_idx = e_idx if e_idx < len(x) else len(x)
            x_batch = x[s_idx: e_idx]
            if y is not None:
                y_batch = y[s_idx: e_idx]
                yield x_batch, y_batch
            else:
                yield x_batch

    def fit(self, x, y, batch_size=64, epochs=1, validation_data=None):
        for epoch in range(1, epochs + 1):
            print('Epoch {}/{}'.format(epoch, epochs), flush=True)

            for x_batch, y_batch in self.next_batch(x, y, batch_size=batch_size, shuffle=True):
                feed_data = {
                    self.text_input: x_batch,
                    self.label_input: y_batch,
                    self.feature_keep_prob_input: self.feature_keep_prob
                }
                _, loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_data)
                loss = float(loss)
                print('\r - loss:{}'.format(loss), end='', flush=True)
            print()

            if validation_data is not None:
                metrics = self._evaluate_on_val_data(validation_data)
                print(' - val_acc:{:.4f} - val_f1:{:.4f}'.format(metrics['acc'], metrics['f1']))

    def _evaluate_on_val_data(self, validation_data):
        metrics = dict()
        val_pred = self.predict(validation_data[0])
        metrics['acc'] = accuracy_score(validation_data[-1], val_pred)
        metrics['f1'] = f1_score(validation_data[-1], val_pred)
        return metrics

    def predict(self, x, batch_size=64):
        result = []
        for x_batch in self.next_batch(x, batch_size=batch_size):
            feed_data = {
                self.text_input: x_batch,
                self.feature_keep_prob_input: 1.0
            }
            res = self.sess.run(self.probs, feed_dict=feed_data)
            result.append(res)
        result = np.concatenate(result, axis=0)
        result = np.where(result > 0.5, 1, 0)
        return result

    def predict_prob(self, x):
        result = []
        for x_batch in self.next_batch(x, batch_size=batch_size):
            feed_data = {
                self.text_input: x_batch,
                self.feature_keep_prob_input: 1.0
            }
            res = self.sess.run(self.probs, feed_dict=feed_data)
            result.append(res)
        result = np.concatenate(result, axis=0)
        return result
