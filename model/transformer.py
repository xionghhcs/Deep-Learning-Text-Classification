import tensorflow as tf

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


class Transformer(object):
    def __init__(self, matrix, maxlen, num_classes=2, d_model=256, d_inner_hid=512, n_head=4, d_k=64, d_v=64, layers=2,
                 dropout=0.1, lr=0.001):
        self.matrix = matrix
        self.maxlen = maxlen
        self.num_classes = num_classes
        self.d_model = d_model
        self.d_inner_hid = d_inner_hid
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.layers = layers
        self.dropout = dropout
        self.lr = lr

        self.build_model()
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def build_model(self):
        with tf.variable_scope('inputs'):
            self.text_input = tf.placeholder(dtype=tf.int32, shape=[None, self.maxlen], name='text_input')
            self.label_input = tf.placeholder(dtype=tf.int32, shape=[None, ], name='label_input')

            self.attn_dropout_input = tf.placeholder_with_default(0.0, shape=(), name='attn_dropout_input')
            self.train_flag = tf.placeholder(dtype=tf.bool)

        with tf.variable_scope('pos_seq'):
            pos_input = self._get_pos_seq(self.text_input)

        with tf.variable_scope('embeddings'):
            word_embedding = tf.get_variable(name='word_embed', shape=self.matrix.shape,
                                             initializer=tf.constant_initializer(self.matrix), trainable=False)

            pos_matrix = self._get_pos_embedding_matrix(max_len=self.maxlen, embed_dim=self.d_model)

            pos_embedding = tf.get_variable(name='pos_embed', shape=pos_matrix.shape,
                                            initializer=tf.constant_initializer(pos_matrix), trainable=False)

        with tf.variable_scope('encoder'):
            text_encoded = self._encoder(self.text_input, word_embed=word_embedding, pos_embed=pos_embedding,
                                         layers=self.layers, return_attn=False)

            text_rep = tf.reduce_mean(text_encoded, axis=1)

        with tf.variable_scope('output'):
            dense = text_rep
            out_ = tf.layers.dense(dense, units=self.num_classes, activation=None)

            self.probs = tf.nn.softmax(out_, axis=-1)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out_, labels=self.label_input))

        with tf.variable_scope('train_op'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def _get_pad_mask(self, s, k):
        ones = tf.expand_dims(tf.ones_like(s, 'float32'), -1)
        mask = tf.cast(tf.expand_dims(tf.not_equal(k, 0), 1), 'float32')
        mask = tf.matmul(ones, mask)
        return mask

    def _get_pos_seq(self, x):
        mask = tf.cast(tf.not_equal(x, 0), 'int32')
        pos = tf.cumsum(tf.ones_like(x, 'int32'), 1)
        return pos * mask

    def _encoder(self, src_seq, word_embed, pos_embed, layers, return_attn=False):
        x = tf.nn.embedding_lookup(word_embed, src_seq)
        src_pos = self._get_pos_seq(src_seq)
        pos = tf.nn.embedding_lookup(pos_embed, src_pos)
        x = tf.add(x, pos)

        if return_attn:
            atts = []

        mask = self._get_pad_mask(src_seq, src_seq)

        for idx in range(layers):
            with tf.variable_scope('encoder_layer_{}'.format(idx)):
                x, attn = self._encoder_layer(x, self.d_model, self.d_inner_hid, self.n_head, self.d_k, self.d_v,
                                              dropout=self.attn_dropout_input, mask=mask)
                if return_attn:
                    atts.append(attn)

        return (x, atts) if return_attn else x

    def _encoder_layer(self, x, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1, mask=None):

        output, attn = self._multihead_attention(x, x, x, d_model, n_head, d_k, d_v, dropout, mask)

        output = self._feedforward(output, d_model, d_inner_hid, dropout=dropout)

        return output, attn

    def _multihead_attention(self, q, k, v, d_model, n_head, d_k, d_v, dropout=0.1, mask=None, use_layer_norm=True):
        def reshape1(x):
            s = tf.shape(x)
            x = tf.reshape(x, [s[0], s[1], n_head, d_k])
            x = tf.transpose(x, [2, 0, 1, 3])
            x = tf.reshape(x, [-1, s[1], d_k])
            return x

        def reshape2(x):
            s = tf.shape(x)
            x = tf.reshape(x, [n_head, -1, s[1], s[2]])
            x = tf.transpose(x, [1, 2, 0, 3])
            x = tf.reshape(x, [-1, s[1], n_head * d_v])
            return x

        qs = tf.layers.dense(q, units=n_head * d_k, use_bias=False)
        ks = tf.layers.dense(k, units=n_head * d_k, use_bias=False)
        vs = tf.layers.dense(v, units=n_head * d_v, use_bias=False)

        qs = reshape1(qs)
        ks = reshape1(ks)
        vs = reshape1(vs)

        if mask is not None:
            mask = tf.tile(mask, multiples=[n_head, 1, 1])

        head, attn = self._scale_dot_product_attention(qs, ks, vs, mask, d_model, dropout)

        head = reshape2(head)

        # in order to feed the output to the next layer, the output shape must be the same as input shape
        output = tf.layers.dense(head, units=d_model)

        output = tf.layers.dropout(output, rate=dropout)

        if not use_layer_norm: return output, attn

        with tf.variable_scope('multihead_attention'):
            output = self._layer_normalization(output)

        return output, attn

    def _feedforward(self, x, d_model, d_inner_hid, dropout=0.1):
        output = tf.layers.conv1d(x, d_inner_hid, 1, activation=tf.nn.relu)
        output = tf.layers.conv1d(output, d_model, 1)
        output = tf.layers.dropout(output, rate=dropout, training=self.train_flag)
        output = tf.add(output, x)
        with tf.variable_scope('feedforward'):
            output = self._layer_normalization(output)
        return output

    def _scale_dot_product_attention(self, q, k, v, mask, d_model, attn_dropout=0.1):
        temper = np.sqrt(d_model)

        attn = tf.matmul(q, tf.transpose(k, [0, 2, 1])) / temper

        if mask is not None:
            mmask = (-1e+10) * (1 - mask)
            attn = tf.add(attn, mask)

        attn = tf.nn.softmax(attn)
        attn = tf.layers.dropout(attn, rate=attn_dropout, training=self.train_flag)
        output = tf.matmul(attn, v)

        return output, attn

    def _layer_normalization(self, x, eps=1e-6):
        mean, var = tf.nn.moments(x, axes=[2], keep_dims=True)
        std = tf.sqrt(var)

        gamma = tf.get_variable(name='gamma', shape=x.shape.as_list()[-1:])
        beta = tf.get_variable(name='beta', shape=x.shape.as_list()[-1:])
        return gamma * (x - mean) / (std + eps) + beta

    def _get_pos_embedding_matrix(self, max_len, embed_dim):
        pos_matrix = np.array([
            [pos / np.power(10000, 2 * (j // 2) / embed_dim) for j in range(embed_dim)]
            if pos != 0 else np.zeros(embed_dim)
            for pos in range(max_len)
        ])
        pos_matrix[1:, 0::2] = np.sin(pos_matrix[1:, 0::2])
        pos_matrix[1:, 1::2] = np.cos(pos_matrix[1:, 1::2])
        return pos_matrix

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
                    self.attn_dropout_input: 0.1,
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
                self.train_flag: False
            }
            res = self.sess.run(self.probs, feed_dict=feed_data)
            result.append(res)
        result = np.concatenate(result, axis=0)
        return result
