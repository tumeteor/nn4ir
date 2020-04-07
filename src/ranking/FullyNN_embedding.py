import tensorflow as tf
from Util.configs import NNConfig
from Util.dataloader import DataLoader
from argparse import ArgumentParser
from src.ranking.NN import NN
import numpy as np


class FENN(NN):
    def __init__(self, embedding, pretrained, qsize="1k"):
        super(FENN, self).__init__()
        self.embedded_train_expanded = None
        self.embedded_valid_expanded = None
        self.embedded_test_expanded = None
        self.mode = None
        self.cache = None
        self.pretrained = pretrained
        self.d_loader = DataLoader(embedding=embedding, pretrained=pretrained, qsize=qsize)
        self.input_vector_size = NNConfig.max_doc_size
        # output vector size = 1 for scoring model
        self.output_vector_size = 10 if self.ordinal else 1
        self.train_dataset, self.train_labels, self.train_queries, self.valid_dataset, \
        self.valid_labels, self.valid_queries, \
        self.test_dataset, self.test_labels, self.test_queries = self.d_loader.get_ttv2() if self.cache \
            else self.d_loader.get_ttv()

    def simple_NN_prob(self):
        self.log.info("creating the computational graph...")
        graph = tf.Graph()
        with graph.as_default():
            with tf.device(self.mode):
                # Input data
                tf_train_left = tf.placeholder(tf.int32, shape=(NNConfig.batch_size, self.input_vector_size))
                tf_train_right = tf.placeholder(tf.int32, shape=(NNConfig.batch_size, self.input_vector_size))
                tf_train_labels = tf.placeholder(tf.float32, shape=(NNConfig.batch_size, self.output_vector_size))
                tf_train_queries = tf.placeholder(tf.int32, shape=(NNConfig.batch_size, self.input_vector_size))

                tf_valid_left = tf.placeholder(tf.int32, shape=(NNConfig.batch_size, self.input_vector_size))
                tf_valid_right = tf.placeholder(tf.int32, shape=(NNConfig.batch_size, self.input_vector_size))
                tf_valid_queries = tf.placeholder(tf.int32, shape=(NNConfig.batch_size, self.input_vector_size))

                tf_test_left = tf.placeholder(tf.int32, shape=(NNConfig.batch_size, self.input_vector_size),
                                              name="test_left")
                tf_test_right = tf.placeholder(tf.int32, shape=(NNConfig.batch_size, self.input_vector_size),
                                               name="test_right")
                tf_test_queries = tf.placeholder(tf.int32, shape=(NNConfig.batch_size, self.input_vector_size),
                                                 name="test_queries")

                if NNConfig.regularization:
                    beta_regu = tf.placeholder(tf.float32)

                if NNConfig.learning_rate_decay:
                    global_step = tf.Variable(0)

                # Variables.
                def init_weights(shape):
                    return tf.Variable(tf.truncated_normal(shape))

                def init_biases(shape):
                    return tf.Variable(tf.zeros(shape))

                w_o = init_weights([NNConfig.num_hidden_nodes, self.output_vector_size])
                b_o = init_biases([self.output_vector_size])

                # Embedding layer
                with tf.device('/gpu:0'), tf.name_scope("embedding"):

                    W = self.embedding_W()
                    embedded_left = tf.nn.embedding_lookup(W, tf_train_left)
                    self.embedded_train_left = tf.reduce_sum(embedded_left, [1])

                    embedded_right = tf.nn.embedding_lookup(W, tf_train_right)
                    self.embedded_train_right = tf.reduce_sum(embedded_right, [1])

                    embedded_queries = tf.nn.embedding_lookup(W, tf_train_queries)
                    self.embedded_train_queries = tf.reduce_sum(embedded_queries, [1])

                    embedded_valid_left = tf.nn.embedding_lookup(W, tf_valid_left)
                    self.embedded_valid_left = tf.reduce_sum(embedded_valid_left, [1])

                    embedded_valid_right = tf.nn.embedding_lookup(W, tf_valid_right)
                    self.embedded_valid_right = tf.reduce_sum(embedded_valid_right, [1])

                    embedded_valid_queries = tf.nn.embedding_lookup(W, tf_valid_queries)
                    self.embedded_valid_queries = tf.reduce_sum(embedded_valid_queries, [1])

                    embedded_test = tf.nn.embedding_lookup(W, tf_test_left)
                    self.embedded_test_left = tf.reduce_sum(embedded_test, [1])

                    embedded_test = tf.nn.embedding_lookup(W, tf_test_right)
                    self.embedded_test_right = tf.reduce_sum(embedded_test, [1])

                    embedded_test_queries = tf.nn.embedding_lookup(W, tf_test_queries)
                    self.embedded_test_queries = tf.reduce_sum(embedded_test_queries, [1])

                # Training computation
                def model(data_left, data_right, queries, w_o, b_o, train_mode, name=None):
                    dataset = tf.concat([queries, data_left, data_right], axis=1)
                    w_hs = []
                    if NNConfig.dropout and train_mode:
                        drop_h = dataset
                        for i in range(0, NNConfig.num_hidden_layers):
                            drop_i = tf.nn.dropout(drop_h, NNConfig.dropout_keep_prob_input)
                            w_h = init_weights([drop_h.shape.as_list()[1], NNConfig.num_hidden_nodes])
                            b_h = init_biases([NNConfig.num_hidden_nodes])

                            h_lay_train = tf.nn.relu(tf.matmul(drop_i, w_h) + b_h)
                            drop_h = tf.nn.dropout(h_lay_train, NNConfig.dropout_keep_prob_hidden)
                            w_hs.append(w_h)

                        return tf.add(tf.matmul(drop_h, w_o), b_o, name=name), w_hs

                    else:
                        h_lay_train = dataset
                        for i in range(0, NNConfig.num_hidden_layers):
                            w_h = init_weights([h_lay_train.shape.as_list()[1], NNConfig.num_hidden_nodes])
                            b_h = init_biases([NNConfig.num_hidden_nodes])
                            h_lay_train = tf.nn.relu(tf.matmul(h_lay_train, w_h) + b_h)  # or tf.nn.sigmoid
                            w_hs.append(w_h)

                        return tf.add(tf.matmul(h_lay_train, w_o), b_o, name=name), w_hs

                logits, w_hs = model(self.embedded_train_left, self.embedded_train_right, self.embedded_train_queries,
                                     w_o, b_o, True)

                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))

                self.log.info("embedded_train shape: {}".format(tf.shape(self.embedded_train_left)))

                if NNConfig.regularization:
                    loss += beta_regu * (sum(tf.nn.l2_loss(w_h) for w_h in w_hs) + tf.nn.l2_loss(w_o))
                if NNConfig.learning_rate_decay:
                    learning_rate = tf.train.exponential_decay(NNConfig.learning_rate, global_step,
                                                               NNConfig.decay_steps, NNConfig.decay_rate,
                                                               staircase=True)
                    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
                else:
                    optimizer = tf.train.AdamOptimizer(NNConfig.learning_rate).minimize(loss)

                # score model: linear activation
                train_prediction = logits
                valid_prediction, wv = model(self.embedded_valid_left, self.embedded_valid_right,
                                             self.embedded_valid_queries, w_o, b_o, False)
                with tf.name_scope("test_prediction"):
                    test_prediction, tv = model(self.embedded_test_left, self.embedded_test_right,
                                                self.embedded_test_queries, w_o, b_o, False, name="test_prediction")

                '''
                run accuracy scope
                '''
                with tf.name_scope('accuracy'):
                    pre = tf.placeholder("float", shape=[None, self.output_vector_size])
                    lbl = tf.placeholder("float", shape=[None, self.output_vector_size])
                    # compute the mean of all predictions
                    accuracy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pre, labels=lbl))

        self.log.info('running the session...')

        self.train_pairwise(graph, self.train_dataset, self.train_labels, self.train_queries,
                            self.valid_dataset, self.valid_labels, self.valid_queries, self.test_dataset,
                            self.test_labels, self.test_queries,
                            tf_train_left, tf_train_right,
                            tf_train_labels, tf_train_queries, tf_valid_left, tf_valid_right, tf_valid_queries,
                            tf_test_left, tf_test_right, tf_test_queries,
                            train_prediction, valid_prediction,
                            test_prediction, loss, optimizer, accuracy, pre, lbl, beta_regu, prob=True)

    def simple_NN_pairwise(self):
        self.log.info("creating the computational graph...")
        graph = tf.Graph()
        with graph.as_default():
            with tf.device(self.mode):
                # Input data
                tf_train_left = tf.placeholder(tf.int32, shape=(NNConfig.batch_size, self.input_vector_size))
                tf_train_right = tf.placeholder(tf.int32, shape=(NNConfig.batch_size, self.input_vector_size))
                tf_train_labels = tf.placeholder(tf.float32, shape=(NNConfig.batch_size, self.output_vector_size))
                tf_train_queries = tf.placeholder(tf.int32, shape=(NNConfig.batch_size, self.input_vector_size))

                tf_valid_left = tf.placeholder(tf.int32, shape=(NNConfig.batch_size, self.input_vector_size))
                tf_valid_right = tf.placeholder(tf.int32, shape=(NNConfig.batch_size, self.input_vector_size))
                tf_valid_queries = tf.placeholder(tf.int32, shape=(NNConfig.batch_size, self.input_vector_size))

                tf_test_left = tf.placeholder(tf.int32, shape=(NNConfig.batch_size, self.input_vector_size),
                                              name="test_left")
                tf_test_right = tf.placeholder(tf.int32, shape=(NNConfig.batch_size, self.input_vector_size),
                                               name="test_right")
                tf_test_queries = tf.placeholder(tf.int32, shape=(NNConfig.batch_size, self.input_vector_size),
                                                 name="test_queries")

                if NNConfig.regularization:
                    beta_regu = tf.placeholder(tf.float32)

                if NNConfig.learning_rate_decay:
                    global_step = tf.Variable(0)

                # Variables.
                def init_weights(shape):
                    # return tf.Variable(tf.random_normal(shape, stddev=0.01))
                    return tf.Variable(tf.truncated_normal(shape))

                def init_biases(shape):
                    return tf.Variable(tf.zeros(shape))

                w_o = init_weights([NNConfig.num_hidden_nodes, self.output_vector_size])
                b_o = init_biases([self.output_vector_size])

                # Embedding layer
                with tf.device('/cpu:0'), tf.name_scope("embedding"):

                    W = self.embedding_W()
                    embedded_left = tf.nn.embedding_lookup(W, tf_train_left)
                    self.embedded_train_left = tf.reduce_sum(embedded_left, [1])

                    embedded_right = tf.nn.embedding_lookup(W, tf_train_right)
                    self.embedded_train_right = tf.reduce_sum(embedded_right, [1])

                    embedded_queries = tf.nn.embedding_lookup(W, tf_train_queries)
                    self.embedded_train_queries = tf.reduce_sum(embedded_queries, [1])

                    embedded_valid_left = tf.nn.embedding_lookup(W, tf_valid_left)
                    self.embedded_valid_left = tf.reduce_sum(embedded_valid_left, [1])

                    embedded_valid_right = tf.nn.embedding_lookup(W, tf_valid_right)
                    self.embedded_valid_right = tf.reduce_sum(embedded_valid_right, [1])

                    embedded_valid_queries = tf.nn.embedding_lookup(W, tf_valid_queries)
                    self.embedded_valid_queries = tf.reduce_sum(embedded_valid_queries, [1])

                    embedded_test = tf.nn.embedding_lookup(W, tf_test_left)
                    self.embedded_test_left = tf.reduce_sum(embedded_test, [1])

                    embedded_test = tf.nn.embedding_lookup(W, tf_test_right)
                    self.embedded_test_right = tf.reduce_sum(embedded_test, [1])

                    embedded_test_queries = tf.nn.embedding_lookup(W, tf_test_queries)
                    self.embedded_test_queries = tf.reduce_sum(embedded_test_queries, [1])

                # Training computation
                def model(dataset, queries, w_o, b_o, train_mode):
                    dataset = tf.concat([queries, dataset], axis=1)
                    w_hs = []
                    if NNConfig.dropout and train_mode:
                        drop_h = dataset
                        for i in range(0, NNConfig.num_hidden_layers):
                            drop_i = tf.nn.dropout(drop_h, NNConfig.dropout_keep_prob_input)
                            w_h = init_weights([drop_h.shape.as_list()[1], NNConfig.num_hidden_nodes])
                            b_h = init_biases([NNConfig.num_hidden_nodes])

                            h_lay_train = tf.nn.relu(tf.matmul(drop_i, w_h) + b_h)
                            drop_h = tf.nn.dropout(h_lay_train, NNConfig.dropout_keep_prob_hidden)
                            w_hs.append(w_h)

                        return tf.matmul(drop_h, w_o) + b_o, w_hs
                    else:
                        h_lay_train = dataset
                        for i in range(0, NNConfig.num_hidden_layers):
                            w_h = init_weights([h_lay_train.shape.as_list()[1], NNConfig.num_hidden_nodes])
                            b_h = init_biases([NNConfig.num_hidden_nodes])
                            h_lay_train = tf.nn.relu(tf.matmul(h_lay_train, w_h) + b_h)  # or tf.nn.sigmoid
                            w_hs.append(w_h)

                        return tf.matmul(h_lay_train, w_o) + b_o, w_hs

                def pairwise_model(data_left, data_right, queries, w_o, b_o, train_mode):
                    logits_left, w_hs_left = model(data_left, queries, w_o, b_o, train_mode)
                    logits_right, w_hs_right = model(data_right, queries, w_o, b_o, train_mode)

                    logits = logits_left - logits_right
                    return logits, w_hs_left + w_hs_right

                logits, w_hs = pairwise_model(self.embedded_train_left, self.embedded_train_right,
                                              self.embedded_train_queries,
                                              w_o, b_o, True)

                loss = tf.losses.hinge_loss(labels=tf_train_labels, logits=logits)

                self.log.info("embedded_train shape: {}".format(tf.shape(self.embedded_train_left)))

                if NNConfig.regularization:
                    loss += beta_regu * (sum(tf.nn.l2_loss(w_h) for w_h in w_hs) + tf.nn.l2_loss(w_o))
                if NNConfig.learning_rate_decay:
                    learning_rate = tf.train.exponential_decay(NNConfig.learning_rate, global_step,
                                                               NNConfig.decay_steps, NNConfig.decay_rate,
                                                               staircase=True)
                    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
                else:
                    optimizer = tf.train.AdamOptimizer(NNConfig.learning_rate).minimize(loss)

                # score model: linear activation
                train_prediction = logits
                valid_prediction, w_v = pairwise_model(self.embedded_valid_left, self.embedded_valid_right,
                                                       self.embedded_valid_queries, w_o, b_o, False)
                with tf.name_scope("test_prediction"):
                    test_prediction, tv = model(self.embedded_test_left, self.embedded_test_right,
                                                self.embedded_test_queries, w_o, b_o, False)
                    test_prediction = tf.identity(test_prediction, name="test_prediction")

                '''
                run accuracy scope
                '''
                with tf.name_scope('accuracy'):
                    pre = tf.placeholder("float", shape=[None, self.output_vector_size])
                    lbl = tf.placeholder("float", shape=[None, self.output_vector_size])
                    # compute the mean of all predictions
                    accuracy = tf.losses.hinge_loss(labels=lbl, logits=pre)

        self.log.info('running the session...')

        self.train_pairwise(graph, self.train_dataset, self.train_labels, self.valid_dataset, self.valid_labels,
                            self.test_dataset, self.test_labels,
                            tf_train_left, tf_train_right,
                            tf_train_labels, tf_valid_left, tf_valid_right, tf_test_left, tf_test_right,
                            train_prediction, valid_prediction,
                            test_prediction, loss, optimizer, accuracy, pre, lbl, beta_regu)

    def simple_NN(self):
        self.log.info("creating the computational graph...")
        graph = tf.Graph()
        with graph.as_default():
            with tf.device(self.mode):
                # Input data
                tf_train_dataset = tf.placeholder(tf.int32, shape=(NNConfig.batch_size, self.input_vector_size))
                tf_train_labels = tf.placeholder(tf.float32, shape=(NNConfig.batch_size, self.output_vector_size))

                tf_train_queries = tf.placeholder(tf.int32, shape=(NNConfig.batch_size, self.input_vector_size))

                # create a placeholder
                tf_valid_dataset = tf.placeholder(tf.int32, shape=(NNConfig.batch_size, self.input_vector_size))
                tf_valid_queries = tf.placeholder(tf.int32, shape=(NNConfig.batch_size, self.input_vector_size))

                tf_test_dataset = tf.placeholder(tf.int32, shape=(NNConfig.batch_size, self.input_vector_size),
                                                 name="test")
                tf_test_queries = tf.placeholder(tf.int32, shape=(NNConfig.batch_size, self.input_vector_size),
                                                 name="test_queries")

                if NNConfig.regularization:
                    beta_regu = tf.placeholder(tf.float32)

                if NNConfig.learning_rate_decay:
                    global_step = tf.Variable(0)

                # Variables.
                def init_weights(shape):
                    # return tf.Variable(tf.random_normal(shape, stddev=0.01))
                    return tf.Variable(tf.truncated_normal(shape))

                def init_biases(shape):
                    return tf.Variable(tf.zeros(shape))

                w_o = init_weights([NNConfig.num_hidden_nodes, self.output_vector_size])
                b_o = init_biases([self.output_vector_size])

                '''
                Embedding layer
                '''
                with tf.device('/gpu:0'), tf.name_scope("embedding"):
                    self.embedding_layer(
                        tf_train_dataset,
                        tf_train_queries,
                        tf_valid_dataset,
                        tf_valid_queries,
                        tf_test_dataset,
                        tf_test_queries
                    )

                # Training computation
                def model(dataset, queries, w_o, b_o, train_mode):
                    dataset = tf.concat([queries, dataset], axis=1)
                    w_hs = []
                    if NNConfig.dropout and train_mode:
                        drop_h = dataset
                        for i in range(0, NNConfig.num_hidden_layers):
                            drop_i = tf.nn.dropout(drop_h, NNConfig.dropout_keep_prob_input)
                            w_h = init_weights([drop_h.shape.as_list()[1], NNConfig.num_hidden_nodes])
                            b_h = init_biases([NNConfig.num_hidden_nodes])

                            h_lay_train = tf.nn.relu(tf.matmul(drop_i, w_h) + b_h)
                            drop_h = tf.nn.dropout(h_lay_train, NNConfig.dropout_keep_prob_hidden)
                            w_hs.append(w_h)

                        if self.ordinal:
                            return tf.nn.sigmoid(tf.matmul(drop_h, w_o) + b_o), w_hs
                        else:
                            return tf.matmul(drop_h, w_o + b_o), w_hs
                    else:
                        h_lay_train = dataset
                        for i in range(0, NNConfig.num_hidden_layers):
                            w_h = init_weights([h_lay_train.shape.as_list()[1], NNConfig.num_hidden_nodes])
                            b_h = init_biases([NNConfig.num_hidden_nodes])
                            h_lay_train = tf.nn.relu(tf.matmul(h_lay_train, w_h) + b_h)  # or tf.nn.sigmoid
                            w_hs.append(w_h)

                        if self.ordinal:
                            return tf.nn.sigmoid(tf.matmul(h_lay_train, w_o) + b_o), w_hs
                        else:
                            return tf.matmul(h_lay_train, w_o) + b_o, w_hs

                self.log.info("embedded_train shape: {}".format(tf.shape(self.embedded_train_expanded)))

                logits, w_hs = model(self.embedded_train_expanded, self.embedded_train_queries, w_o, b_o, True)

                if self.ordinal:
                    loss = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
                else:
                    loss = tf.reduce_sum(tf.pow(logits - tf_train_labels, 2)) / (2 * NNConfig.batch_size)
                if NNConfig.regularization:
                    loss += beta_regu * (sum(tf.nn.l2_loss(w_h) for w_h in w_hs) + tf.nn.l2_loss(w_o))
                if NNConfig.learning_rate_decay:
                    learning_rate = tf.train.exponential_decay(NNConfig.learning_rate, global_step,
                                                               NNConfig.decay_steps, NNConfig.decay_rate,
                                                               staircase=True)
                    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
                else:
                    optimizer = tf.train.AdamOptimizer(NNConfig.learning_rate).minimize(loss)

                # score model: linear activation
                train_prediction = logits
                valid_prediction, w_v = model(self.embedded_valid_expanded, self.embedded_valid_queries, w_o, b_o,
                                              False)
                with tf.name_scope("test_prediction"):
                    test_prediction, w_t = model(self.embedded_test_expanded, self.embedded_test_queries, w_o, b_o,
                                                 False)
                    test_prediction = tf.identity(test_prediction, name="test_prediction")

                '''
                run accuracy scope
                '''
                with tf.name_scope('accuracy'):
                    pre = tf.placeholder("float", shape=[None, self.output_vector_size])
                    lbl = tf.placeholder("float", shape=[None, self.output_vector_size])
                    # compute the mean of all predictions
                    if self.ordinal:
                        accuracy = tf.reduce_mean(
                            tf.cast(tf.nn.sigmoid_cross_entropy_with_logits(logits=pre, labels=lbl), "float"))
                    else:
                        accuracy = tf.reduce_sum(tf.pow(pre - lbl, 2)) / (2 * tf.cast(tf.shape(lbl)[0], tf.float32))

        self.log.info('running the session...')
        self.train(graph, tf_train_dataset, tf_train_labels, tf_train_queries,
                   tf_valid_dataset, tf_valid_queries, tf_test_dataset, tf_test_queries,
                   train_prediction, valid_prediction,
                   test_prediction, loss, optimizer, accuracy, pre, lbl, beta_regu)

    def embedding_W(self):
        if not self.pretrained:
            # Embedding layer
            with tf.device('/cpu:0'), tf.name_scope("embedding"):
                W = tf.Variable(
                    tf.random_uniform([self.d_loader.d_handler.get_vocab_size(), NNConfig.embedding_dim], -1.0, 1.0),
                    name="W")

        elif self.hybrid:
            only_in_train = self.d_loader.d_handler.get_vocab() - self.d_loader.pretrain_vocab
            train_embeddings = tf.get_variable(
                name="embs_only_in_train",
                shape=[len(only_in_train), NNConfig.embedding_dim],
                initializer=tf.random_uniform_initializer(-1.0, 1.0),
                trainable=True)
            embed_vocab_size = len(self.d_loader.pretrain_vocab)
            embedding_dim = len(self.d_loader.embd[0])
            pretrained_embs = tf.Variable(tf.constant(0.0, shape=[embed_vocab_size, embedding_dim]),
                                          trainable=False, name="pretrained")

            self.embedding_placeholder = tf.placeholder(tf.float32, [embed_vocab_size, NNConfig.embedding_dim])
            self.embedding_init = pretrained_embs.assign(self.embedding_placeholder)

            W = tf.concat([pretrained_embs, train_embeddings], axis=0)
        else:
            embed_vocab_size = len(self.d_loader.pretrain_vocab)
            embedding_dim = len(self.d_loader.embd[0])
            self.embedding = np.asarray(self.d_loader.embd)

            W = tf.Variable(tf.constant(0.0, shape=[embed_vocab_size, embedding_dim]),
                            trainable=False, name="W")

            self.embedding_placeholder = tf.placeholder(tf.float32, [embed_vocab_size, NNConfig.embedding_dim])
            self.embedding_init = W.assign(self.embedding_placeholder)

        return W

    def embedding_layer(self, tf_train_dataset, tf_train_queries, tf_valid_dataset, tf_valid_queries,
                        tf_test_dataset, tf_test_queries):
        if not self.pretrained:
            # Embedding layer
            with tf.device('/cpu:0'), tf.name_scope("embedding"):
                self.W = tf.Variable(
                    tf.random_uniform([self.d_loader.d_handler.get_vocab_size(), NNConfig.embedding_dim], -1.0, 1.0),
                    name="W")
                embedded_train = tf.nn.embedding_lookup(self.W, tf_train_dataset)
                self.embedded_train_expanded = tf.reduce_sum(embedded_train, [1])

                embedded_train_queries = tf.nn.embedding_lookup(self.W, tf_train_queries)
                self.embedded_train_queries = tf.reduce_sum(embedded_train_queries, [1])

                embedded_valid = tf.nn.embedding_lookup(self.W, tf_valid_dataset)
                self.embedded_valid_expanded = tf.reduce_sum(embedded_valid, [1])

                embedded_valid_queries = tf.nn.embedding_lookup(self.W, tf_valid_queries)
                self.embedded_valid_queries = tf.reduce_sum(embedded_valid_queries, [1])

                embedded_test = tf.nn.embedding_lookup(self.W, tf_test_dataset)
                self.embedded_test_expanded = tf.reduce_sum(embedded_test, [1])

                embedded_test_queries = tf.nn.embedding_lookup(self.W, tf_test_queries)
                self.embedded_test_queries = tf.reduce_sum(embedded_test_queries, [1])
        elif self.hybrid:
            only_in_train = self.d_loader.d_handler.get_vocab() - self.d_loader.pretrain_vocab
            train_embeddings = tf.get_variable(
                name="embs_only_in_train",
                shape=[len(only_in_train), NNConfig.embedding_dim],
                initializer=tf.random_uniform_initializer(-1.0, 1.0),
                trainable=True)
            embed_vocab_size = len(self.d_loader.pretrain_vocab)
            embedding_dim = len(self.d_loader.embd[0])
            pretrained_embs = tf.Variable(tf.constant(0.0, shape=[embed_vocab_size, embedding_dim]),
                                          trainable=False, name="pretrained")

            self.embedding_placeholder = tf.placeholder(tf.float32, [embed_vocab_size, NNConfig.embedding_dim])
            self.embedding_init = pretrained_embs.assign(self.embedding_placeholder)

            W = tf.concat([pretrained_embs, train_embeddings], axis=0)

            # Reduce along dimension 1 (`n_input`) to get a single vector (row)
            # per input example. It's fairly typical to do this for bag-of-words type problems.

            self.embedded_train_expanded = tf.reduce_sum(tf.nn.embedding_lookup(W, tf_train_dataset), [1])
            self.embedded_train_queries = tf.reduce_sum(tf.nn.embedding_lookup(W, tf_train_queries), [1])
            self.embedded_valid_expanded = tf.reduce_sum(tf.nn.embedding_lookup(W, tf_valid_dataset), [1])
            self.embedded_valid_queries = tf.reduce_sum(tf.nn.embedding_lookup(W, tf_valid_queries), [1])
            self.embedded_test_expanded = tf.reduce_sum(tf.nn.embedding_lookup(W, tf_test_dataset), [1])
            self.embedded_test_queries = tf.reduce_sum(tf.nn.embedding_lookup(W, tf_test_queries), [1])

        else:
            embed_vocab_size = len(self.d_loader.pretrain_vocab)
            embedding_dim = len(self.d_loader.embd[0])
            self.embedding = np.asarray(self.d_loader.embd)

            W = tf.Variable(tf.constant(0.0, shape=[embed_vocab_size, embedding_dim]),
                            trainable=False, name="W")

            self.embedding_placeholder = tf.placeholder(tf.float32, [embed_vocab_size, NNConfig.embedding_dim])
            self.embedding_init = W.assign(self.embedding_placeholder)

            # Reduce along dimension 1 (`n_input`) to get a single vector (row)
            # per input example. It's fairly typical to do this for bag-of-words type problems.

            self.embedded_train_expanded = tf.reduce_sum(tf.nn.embedding_lookup(W, tf_train_dataset), [1])
            self.embedded_train_queries = tf.reduce_sum(tf.nn.embedding_lookup(W, tf_train_queries), [1])
            self.embedded_valid_expanded = tf.reduce_sum(tf.nn.embedding_lookup(W, tf_valid_dataset), [1])
            self.embedded_valid_queries = tf.reduce_sum(tf.nn.embedding_lookup(W, tf_valid_queries), [1])
            self.embedded_test_expanded = tf.reduce_sum(tf.nn.embedding_lookup(W, tf_test_dataset), [1])
            self.embedded_test_queries = tf.reduce_sum(tf.nn.embedding_lookup(W, tf_test_queries), [1])


if __name__ == '__main__':
    parser = ArgumentParser(description='Required arguments')
    parser.add_argument('-m', '--mode', help='computation mode', required=False)
    parser.add_argument('-c', '--cache', help='cache data', action='store_true')
    parser.add_argument('-s', '--query_size', help='number of queries for training', required=False)
    parser.add_argument('-l', '--loss', help='loss function [point-wise, pair-wise]', required=False)
    parser.add_argument('-p', '--pretrained', help='pretrained embedding', required=False)
    parser.add_argument('-o')
    args = parser.parse_args()
    if args.pretrained == "pretrained":
        nn = FENN(qsize=args.query_size, pretrained=True, embedding=False)
    else:
        nn = FENN(qsize=args.query_size, pretrained=False, embedding=True)
    try:
        if args.mode == "gpu":
            nn.mode = "/gpu:0"
        else:
            nn.mode = "/cpu:0"
        if args.cache: nn.cache = True

        if args.loss == "point-wise":
            nn.log.info("learn with point-wise loss")
            nn.simple_NN()
        elif args.loss == "pair-wise":
            nn.log.info("learn with pair-wise")
            nn.simple_NN_pairwise()
        else:
            nn.log.info("learn with cross entropy")
            nn.simple_NN_prob()
        nn.log.info("done..")
    except Exception as e:
        nn.log.exception(e)
        raise
