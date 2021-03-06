import collections
import hashlib
import numbers

import tensorflow as tf
import numpy as np
import sys
from sklearn.metrics import f1_score, accuracy_score

try:
    from basic_model import BasicModel
except:
    from .basic_model import BasicModel


class DynamicMemoryNetwork(BasicModel):
    def __init__(self, matrix=None, maxlen=None, q_maxlen=1, num_classes=2, lr=0.001, model_path='../tmp/dmn'):
        super(DynamicMemoryNetwork, self).__init__()

        self.matrix = matrix
        self.maxlen = maxlen
        self.q_maxlen = q_maxlen
        self.num_classes = num_classes

        self.text_encoder_gru_hid = 64

        self.lr = lr
        self.model_path = model_path

        self.build_model()
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def build_model(self):
        with tf.variable_scope('inputs'):
            self.text_input = tf.placeholder(dtype=tf.int32, shape=[None, self.maxlen], name='text_input')
            self.q_input = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='q_input')
            self.label_input = tf.placeholder(dtype=tf.int32, shape=[None, ], name='label')

            self.train_flag = tf.placeholder(dtype=tf.bool, name='train_flag')

        with tf.variable_scope('embedding_layer'):
            embedding = tf.get_variable(
                name='embedding',
                shape=self.matrix.shape,
                initializer=tf.constant_initializer(self.matrix),
                trainable=False
            )
            text_embed = tf.nn.embedding_lookup(embedding, self.text_input)
            q_embed = tf.nn.embedding_lookup(embedding, self.q_input)

        with tf.variable_scope('input_module'):
            gru_cell = tf.nn.rnn_cell.GRUCell(self.text_encoder_gru_hid)

            fact_vec, _ = tf.nn.dynamic_rnn(gru_cell, inputs=text_embed, dtype=tf.float32)

        with tf.variable_scope('question_module'):
            gru_cell = tf.nn.rnn_cell.GRUCell(self.text_encoder_gru_hid)

            _, q_vec = tf.nn.dynamic_rnn(gru_cell, inputs=q_embed, dtype=tf.float32)

        with tf.variable_scope('episodic_memory_module'):
            m = self._episodic_memory_mudole(fact_vec, q_vec)

        with tf.variable_scope('answer_module'):
            final_feature = tf.concat(values=[m, q_vec], axis=-1)

        with tf.variable_scope('output_layer'):
            dense = final_feature
            out_ = tf.layers.dense(dense, units=self.num_classes, activation=None)

            self.probs = tf.nn.softmax(out_, dim=-1)
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_input, logits=out_))

        with tf.variable_scope('train_op'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def _episodic_memory_mudole(self, fact, q_vec):
        fact_list = tf.split(fact, fact.shape[1].value, axis=1)
        fact_list = [tf.squeeze(f, axis=1) for f in fact_list]
        gru = tf.nn.rnn_cell.GRUCell(num_units=q_vec.shape[-1].value)
        m_prev = q_vec
        for c in fact_list:
            g = self._attention_score(c, q_vec, m_prev)
            m_prev = g * gru(c, m_prev)[0] + (1 - g) * m_prev
        return m_prev

    def _attention_score(self, fact, q, m_prev):
        z = tf.concat(values=[fact, m_prev, q, fact * q, fact * m_prev, tf.abs(fact - q), tf.abs(fact - m_prev)],
                      axis=-1)
        o1 = tf.layers.dense(z, units=128, activation=tf.nn.tanh)
        o2 = tf.layers.dense(o1, units=64, activation=tf.nn.tanh)
        return tf.nn.sigmoid(o2)

    def next_batch(self, x, y=None, batch_size=64, shuffle=False):
        if isinstance(x, list):
            data_size = len(x[0])
        else:
            data_size = len(x)
        if shuffle:
            import numpy
            idx = list(range(data_size))
            np.random.shuffle(idx)
            new_x = []
            for i in x:
                new_x.append(i[idx])
            x = new_x
            y = y[idx]

        for idx in range(0, data_size, batch_size):
            s_idx = idx
            e_idx = idx + batch_size
            e_idx = e_idx if e_idx < data_size else data_size
            x_batch = []
            for i in x:
                x_batch.append(i[s_idx: e_idx])
            if y is not None:
                y_batch = y[s_idx: e_idx]
                yield x_batch, y_batch
            else:
                yield x_batch

    def fit(self, x, y, batch_size=64, epochs=1, validation_data=None, save_model=False):

        best_metrics = dict()
        best_metrics['acc'] = 0

        for epoch in range(1, epochs + 1):
            print('Epoch {}/{}'.format(epoch, epochs), flush=True)

            t_loss = 0
            t_acc = 0
            cnt = 0
            for x_batch, y_batch in self.next_batch(x, y, batch_size=batch_size, shuffle=True):
                feed_data = {
                    self.text_input: x_batch[0],
                    self.q_input: x_batch[1],
                    self.label_input: y_batch,
                    # self.embed_dropout_input: self.embed_dropout,
                    # self.dense_dropout_input: self.dense_dropout,
                    self.train_flag: True,
                }
                _, loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_data)
                loss = float(loss)
                cnt += 1
                t_loss += loss
                metrics = self._evaluate_on_val_data(validation_data=(x_batch, y_batch))
                t_acc += metrics['acc']
                print('\r - loss:{:.4f} - acc:{:.4f}'.format(t_loss / cnt, t_acc / cnt), end='', flush=True)
            print()

            if validation_data is not None:
                metrics = self._evaluate_on_val_data(validation_data)
                print(' - val_acc:{:.4f}'.format(metrics['acc']))
                if save_model:
                    if best_metrics['acc'] < metrics['acc']:
                        best_metrics['acc'] = metrics['acc']
                        self.save_weight(self.model_path)

    def _evaluate_on_val_data(self, validation_data):
        metrics = dict()
        val_pred = self.predict(validation_data[0])
        metrics['acc'] = accuracy_score(validation_data[-1], val_pred)
        # metrics['f1'] = f1_score(validation_data[-1], val_pred)
        return metrics

    def predict(self, x, batch_size=64):
        result = []
        for x_batch in self.next_batch(x, batch_size=batch_size):
            feed_data = {
                self.text_input: x_batch[0],
                self.q_input: x_batch[1],
                self.train_flag: False
            }
            res = self.sess.run(self.probs, feed_dict=feed_data)
            result.append(res)
        result = np.concatenate(result, axis=0)
        result = result.argmax(axis=1)
        return result

    def predict_prob(self, x, batch_size=64):
        result = []
        for x_batch in self.next_batch(x, batch_size=batch_size):
            feed_data = {
                self.text_input: x_batch[0],
                self.q_input: x_batch[1],
                self.train_flag: False
            }
            res = self.sess.run(self.probs, feed_dict=feed_data)
            result.append(res)
        result = np.concatenate(result, axis=0)
        return result

