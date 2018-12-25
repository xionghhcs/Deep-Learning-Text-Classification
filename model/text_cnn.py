import tensorflow as tf
import numpy as np
import sys
from sklearn.metrics import f1_score, accuracy_score

try:
    from basic_model import BasicModel
except:
    from .basic_model import BasicModel


class TextCNN(BasicModel):
    def __init__(self, matrix=None, maxlen=None, num_classes=2, filter_sizes=[3, 5, 7], filter_num=64,
                 embed_dropout=0.4, dense_dropout=0.4, lr=0.001, model_path='../tmp/text_cnn'):
        super(TextCNN, self).__init__()

        self.matrix = matrix
        self.maxlen = maxlen
        self.num_classes = num_classes

        self.filter_sizes = filter_sizes
        self.filter_num = filter_num
        self.embed_dropout = embed_dropout
        self.dense_dropout = dense_dropout
        self.lr = lr
        self.model_path = model_path

        self.build_model()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def build_model(self):
        with tf.variable_scope('inputs'):
            self.text_input = tf.placeholder(dtype=tf.int32, shape=[None, self.maxlen], name='text_input')
            self.label_input = tf.placeholder(dtype=tf.int32, shape=[None, ], name='label')

            self.embed_dropout_input = tf.placeholder(dtype=tf.float32, name='embed_dropout_input')
            self.dense_dropout_input = tf.placeholder(dtype=tf.float32, name='dense_dropout_input')
            self.train_flag = tf.placeholder(dtype=tf.bool, name='train_flag')

        with tf.variable_scope('embedding_layer'):
            embedding = tf.get_variable(
                name='embedding',
                shape=self.matrix.shape,
                initializer=tf.constant_initializer(self.matrix),
                trainable=False
            )
            text_embed = tf.nn.embedding_lookup(embedding, self.text_input)
            text_embed = tf.layers.dropout(text_embed, rate=self.embed_dropout_input, training=self.train_flag)

        with tf.variable_scope('cnn_layer'):
            conv_result = []
            for filter_size in self.filter_sizes:
                conv = tf.layers.separable_conv1d(text_embed, filters=self.filter_num, kernel_size=filter_size,
                                                  depth_multiplier=4, activation=tf.nn.relu)
                max_pool = tf.reduce_max(conv, axis=1)
                max_pool = tf.layers.dense(max_pool, units=100, activation=tf.nn.relu)
                max_pool = tf.layers.dropout(max_pool, rate=self.embed_dropout_input, training=self.train_flag)
                conv_result.append(max_pool)

        with tf.variable_scope('output_layer'):
            dense = tf.concat(conv_result, axis=-1)
            # dense = tf.layers.batch_normalization(dense, training=self.train_flag)
            dense = tf.layers.dropout(dense, rate=self.dense_dropout_input, training=self.train_flag)
            out_ = tf.layers.dense(dense, units=self.num_classes, activation=None)

            self.probs = tf.nn.softmax(out_, dim=-1)
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_input, logits=out_))

        with tf.variable_scope('train_op'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

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
                    self.text_input: x_batch,
                    self.label_input: y_batch,
                    self.embed_dropout_input: self.embed_dropout,
                    self.dense_dropout_input: self.dense_dropout,
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
                self.text_input: x_batch,
                self.embed_dropout_input: 0.0,
                self.dense_dropout_input: 0.0,
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
                self.text_input: x_batch,
                self.embed_dropout_input: 0.0,
                self.dense_dropout_input: 0.0,
                self.train_flag: False
            }
            res = self.sess.run(self.probs, feed_dict=feed_data)
            result.append(res)
        result = np.concatenate(result, axis=0)
        return result
