import tensorflow as tf
import numpy as np
import sys
from sklearn.metrics import f1_score, accuracy_score

try:
    from basic_model import BasicModel
except:
    from .basic_model import BasicModel


class VDCNN(BasicModel):
    def __init__(self, maxlen=1024, num_classes=2, num_chars=71, embed_dim=16, k=8, lr=0.001,
                 model_path='../tmp/vdcnn'):
        super(VDCNN, self).__init__()

        self.maxlen = maxlen
        self.num_classes = num_classes
        self.num_chars = num_chars
        self.embed_dim = embed_dim
        self.k = k

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
            self.label_input = tf.placeholder(dtype=tf.int32, shape=[None, ], name='label')

            self.train_flag = tf.placeholder_with_default(False, shape=(), name='train_flag')

        with tf.variable_scope('embedding_layer'):
            embedding = tf.get_variable(
                name='embedding',
                shape=(self.num_chars, self.embed_dim),
                initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01),
                trainable=True
            )
            text_embed = tf.nn.embedding_lookup(embedding, self.text_input)

        with tf.variable_scope('temporal_convolution'):
            conv_output = tf.layers.conv1d(text_embed, filters=64, kernel_size=3, strides=1, padding='same')

        with tf.variable_scope('conv_blocks'):
            x = conv_output
            for feature_size in [64, 128, 256, 512]:
                with tf.variable_scope('convoution_block_{}_{}'.format(3, feature_size)):
                    x = self._conv_block(x, filters=feature_size, kernel_size=3, )
                    x = tf.layers.max_pooling1d(x, pool_size=3, strides=2)

        with tf.variable_scope('kmax_pooling'):
            feature_dim = x.shape[-1].value

            x = self._kmax_pooling(x, k=self.k)
            x = tf.reshape(x, [-1, self.k * feature_dim])

        with tf.variable_scope('output_layer'):
            dense = x
            dense = tf.layers.dense(dense, units=2048, activation=tf.nn.relu)
            dense = tf.layers.dense(dense, units=2048, activation=tf.nn.relu)
            out_ = tf.layers.dense(dense, units=self.num_classes, activation=None)

            self.probs = tf.nn.softmax(out_, dim=-1)
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_input, logits=out_))

        with tf.variable_scope('train_op'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def _kmax_pooling(self, x, k, sorted=True):
        shifted_x = tf.transpose(x, [0, 2, 1])
        top_k = tf.nn.top_k(shifted_x, k=k, sorted=sorted)[0]
        return tf.transpose(top_k, [0, 2, 1])

    def _conv_block(self, x, filters, kernel_size=3, use_bias=False, shortcut=False):
        out1 = tf.layers.conv1d(x, filters=filters, kernel_size=kernel_size, use_bias=use_bias)
        out1 = tf.layers.batch_normalization(out1, training=self.train_flag)
        if shortcut:
            out1 = tf.add(out1, x)
        out1 = tf.nn.relu(out1)

        out2 = tf.layers.conv1d(out1, filters=filters, kernel_size=kernel_size, use_bias=use_bias)
        out2 = tf.layers.batch_normalization(out2, training=self.train_flag)
        if shortcut:
            out2 = tf.add(out2, out1)
        out2 = tf.nn.relu(out2)
        return out2

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
            }
            res = self.sess.run(self.probs, feed_dict=feed_data)
            result.append(res)
        result = np.concatenate(result, axis=0)
        return result

