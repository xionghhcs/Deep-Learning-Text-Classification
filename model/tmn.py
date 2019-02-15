import tensorflow as tf


class TMN:
    def __init__(self, num_classes, vocab_size, seq_len, topic_num, embed_matrix, topic_embed_dim, lr=0.001):
        tf.reset_default_graph()

        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.topic_num = topic_num
        self.embed_matrix = embed_matrix
        self.topic_embed_dim = topic_embed_dim
        self.lr = lr

    def build_model(self):
        with tf.variable_scope('inputs'):
            self.bow_input = tf.placeholder(dtype=tf.int32, shape=(None, self.vocab_size))
            self.seq_input = tf.placeholder(dtype=tf.int32, shape=(None, self.seq_len))
            self.psudo_input = tf.placeholder(dtype=tf.int32, shape=(None, self.topic_num))
            self.label_input = tf.placeholder(dtype=tf.int32, shape=(None,))

            self.train_flag = tf.placeholder_with_default(False, shape=())

        with tf.variable_scope('embeddings'):
            self.topic_embedding = tf.get_variable(name='topic_embedding', shape=(self.topic_num, self.vocab_size),
                                                   initializer=tf.random_uniform_initializer, trainable=True)
            self.seq_embedding = tf.get_variable(name='seq_embedding', shape=self.embed_matrix.shape,
                                                 initializer=tf.constant_initializer(self.embed_matrix),
                                                 trainable=False)

        with tf.variable_scope('ntm'):
            represent, represent_mu, p_x = self.neural_topic_model(self.bow_input)

        with tf.variable_scope('memnet'):
            seq_embed = tf.nn.embedding_lookup(self.seq_embedding, seq_input)
            seq_embed = tf.layers.dense(seq_embed, units=self.topic_embed_dim, activation=tf.nn.relu)

            wt_embed = tf.nn.embedding_lookup(self.topic_embedding, psudo_input)
            wt_embed = tf.layers.dense(wt_embed, units=self.topic_embed_dim, activation=tf.nn.relu)

            topic_sum = self.topic_memnet_module(self.seq_input, self.psudo_input)

        with tf.variable_scope('classifier'):
            topic_sum = tf.reshape(topic_sum, [-1, self.seq_len, self.topic_embed_dim, 1])
            self.logits = self.classifier(topic_sum)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.label_input))
            self.probs = tf.nn.softmax(self.logits)
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def _sampling(self, args):
        mu, log_sigma = args
        epsilon = tf.random_normal(shape=(self.topic_num,), mean=0.0, stddev=1.0)
        return mu + tf.exp(log_sigma / 2) * epsilon

    def neural_topic_model(self, x_bow):
        # encoder
        h = tf.layers.dense(x_bow, units=500, activation=tf.nn.relu)
        h = tf.layers.dense(h, units=500, activation=tf.nn.relu)

        z_mean = tf.layers.dense(h, units=self.topic_num)
        z_log_var = tf.layers.dense(h, units=self.topic_num)

        hidden_z = self._sampling([z_mean, z_log_var])

        # neural topic model generation
        def ntm_generator(h):
            tmp = tf.layers.dense(h, units=self.topic_num)
            tmp = tf.layers.dense(tmp, units=self.topic_num)
            tmp = tf.layers.dense(tmp, units=self.topic_num)
            r = tf.layers.dense(tmp, units=self.topic_num)
            return r

        represent = ntm_generator(hidden_z)
        represent_mu = ntm_generator(z_mean)

        # neural topic model decoder
        p_x = tf.layers.dense(represent, units=self.vocab_size, activation=tf.nn.softmax)

        return represent, represent_mu, p_x

    def topic_memory_network_module(self, x_seq, wt_embed, represent_mu):
        match = tf.einsum('ijk,ijk->ijj', seq_embed, wt_embed)
        rep_mu = tf.expand_dims(represent_mu, axis=1)
        rep_mu = tf.tile(rep_mu, match.get_shape()[1])
        joint_match = tf.add(match, represent_mu)
        joint_match = tf.layers.dense(joint_match, units=self.topic_embed_dim)
        topic_sum = tf.add(seq_embed, joint_match)
        topic_sum = tf.layers.dense(topic_sum, units=self.topic_embed_dim)
        return topic_sum

    def classifier(self, topic_sum):
        filter_sizes = [1, 2, 3]
        num_filters = 512
        x1 = tf.layers.conv2d(topic_sum, filters=num_filters, kernel_size=[filter_sizes[0], self.topic_embed_dim],
                              padding='valid')
        x1 = tf.layers.max_pooling2d(pool_size=[self.seq_len - filter_sizes[0] + 1, 1], strides=(1, 1), padding='valid')
        x2 = tf.layers.conv2d(topic_sum, filters=num_filters, kernel_size=[filter_sizes[1], self.topic_embed_dim],
                              padding='valid')
        x2 = tf.layers.max_pooling2d(pool_size=[self.seq_len - filter_sizes[1] + 1, 1], strides=(1, 1), padding='valid')
        x3 = tf.layers.conv2d(topic_sum, filters=num_filters, kernel_size=[filter_sizes[2], self.topic_embed_dim],
                              padding='valid')
        x3 = tf.layers.max_pooling2d(pool_size=[self.seq_len - filter_sizes[2] + 1, 1], strides=(1, 1), padding='valid')

        out = tf.concat(values=[x1, x2, x3], axis=1)
        out = tf.layers.dropout(out, rate=0.05, training=self.train_flag)
        out = tf.layers.dense(self.num_classes, activation=None)
        return out

