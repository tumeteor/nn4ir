import tensorflow as tf
from Util.configs import NNConfig
from Util.dataloader import DataLoader
from argparse import ArgumentParser
from ranking.NN import NN
import numpy as np



class NN(NN):
    def __init__(self, qsize="1k"):
        self.d_loader = DataLoader(embedding=True, qsize=qsize)
        self.input_vector_size = NNConfig.max_doc_size
        super(NN, self).__init__()
        # output vector size = 1 for scoring model
        self.output_vector_size = 1
        self.train_dataset, self.train_labels, self.valid_dataset, \
        self.valid_labels, self.test_dataset, self.test_labels = self.d_loader.get_ttv()

        self.embedded_train_expanded = None
        self.embedded_valid_expanded = None
        self.embedded_test_expanded = None
        self.mode = None


    def simple_NN_prob(self):
        self.log.info("creating the computational graph...")
        graph = tf.Graph()
        with graph.as_default():
            with tf.device(self.mode):
                # Input data
                tf_train_left = tf.placeholder(tf.int32, shape=(NNConfig.batch_size, self.input_vector_size))
                tf_train_right = tf.placeholder(tf.int32, shape=(NNConfig.batch_size, self.input_vector_size))
                tf_train_labels = tf.placeholder(tf.float32, shape=(NNConfig.batch_size, self.output_vector_size))

                tf_valid_left = tf.placeholder(tf.int32, shape=(NNConfig.batch_size, self.input_vector_size))
                tf_valid_right = tf.placeholder(tf.int32, shape=(NNConfig.batch_size, self.input_vector_size))

                tf_test_left = tf.placeholder(tf.int32, shape=(NNConfig.batch_size, self.input_vector_size))
                tf_test_right = tf.placeholder(tf.int32, shape=(NNConfig.batch_size, self.input_vector_size))

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

                w_h = init_weights([NNConfig.embedding_dim, NNConfig.num_hidden_nodes])
                b_h = init_biases([NNConfig.num_hidden_nodes])

                w_o = init_weights([NNConfig.num_hidden_nodes, self.output_vector_size])
                b_o = init_biases([self.output_vector_size])

                # Embedding layer
                with tf.device('/cpu:0'), tf.name_scope("embedding"):
                    self.W = tf.Variable(
                        tf.random_uniform([self.d_loader.d_handler.get_vocab_size(), NNConfig.embedding_dim], -1.0,
                                          1.0),
                        name="W")
                    embedded_left = tf.nn.embedding_lookup(self.W, tf_train_left)
                    self.embedded_train_left = tf.reduce_sum(embedded_left, [1])

                    embedded_right = tf.nn.embedding_lookup(self.W, tf_train_right)
                    self.embedded_train_right = tf.reduce_sum(embedded_right, [1])

                    embedded_valid = tf.nn.embedding_lookup(self.W, tf_valid_left)
                    self.embedded_valid_left = tf.reduce_sum(embedded_valid, [1])

                    embedded_valid = tf.nn.embedding_lookup(self.W, tf_valid_right)
                    self.embedded_valid_right = tf.reduce_sum(embedded_valid, [1])

                    embedded_test = tf.nn.embedding_lookup(self.W, tf_test_left)
                    self.embedded_test_left = tf.reduce_sum(embedded_test, [1])

                    embedded_test = tf.nn.embedding_lookup(self.W, tf_test_right)
                    self.embedded_test_right = tf.reduce_sum(embedded_test, [1])



                # Training computation
                def model(data_left, data_right, w_h, b_h, w_o, b_o, train_mode):
                    dataset = np.concatenate((data_left, data_right), axis=0)
                    if NNConfig.dropout and train_mode:
                        drop_h = dataset
                        for i in range(0, NNConfig.num_hidden_layers):
                            drop_i = tf.nn.dropout(drop_h, NNConfig.dropout_keep_prob_input)
                            w_h = init_weights([drop_h.shape.as_list()[1], NNConfig.num_hidden_nodes])
                            b_h = init_biases([NNConfig.num_hidden_nodes])

                            h_lay_train = tf.nn.relu(tf.matmul(drop_i, w_h) + b_h)
                            drop_h = tf.nn.dropout(h_lay_train, NNConfig.dropout_keep_prob_hidden)

                        return tf.matmul(drop_h, w_o) + b_o
                    else:
                        h_lay_train = dataset
                        for i in range(0, NNConfig.num_hidden_layers):
                            w_h = init_weights([h_lay_train.shape.as_list()[1], NNConfig.num_hidden_nodes])
                            b_h = init_biases([NNConfig.num_hidden_nodes])
                            h_lay_train = tf.nn.relu(tf.matmul(h_lay_train, w_h) + b_h)  # or tf.nn.sigmoid

                        return tf.matmul(h_lay_train, w_o) + b_o


                logits = model(self.embedded_train_left, self.embedded_test_right, w_h, b_h, w_o, b_o, True)

                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, tf_train_labels))

                self.log.info("embedded_train shape: {}".format(tf.shape(self.embedded_train_left)))

                # loss = tf.reduce_sum(tf.pow(logits - tf_train_labels, 2)) / (2 * NNConfig.batch_size)
                # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
                # tf.nn.sigmoid_cross_entropy_with_logits instead of tf.nn.softmax_cross_entropy_with_logits for multi-label case
                if NNConfig.regularization:
                    loss += beta_regu * (tf.nn.l2_loss(w_h) + tf.nn.l2_loss(w_o))
                if NNConfig.learning_rate_decay:
                    learning_rate = tf.train.exponential_decay(NNConfig.learning_rate, global_step,
                                                               NNConfig.decay_steps, NNConfig.decay_rate,
                                                               staircase=True)
                    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
                else:
                    optimizer = tf.train.AdamOptimizer(NNConfig.learning_rate).minimize(loss)
                    # optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(loss)

                # score model: linear activation
                train_prediction = logits
                valid_prediction = model(self.embedded_valid_left, self.embedded_valid_right, w_h, b_h, w_o, b_o, False)
                test_prediction = model(self.embedded_test_left, self.embedded_test_right, w_h, b_h, w_o, b_o, False)

                '''
                run accuracy scope
                '''
                with tf.name_scope('accuracy'):
                    pre = tf.placeholder("float", shape=[None, self.output_vector_size])
                    lbl = tf.placeholder("float", shape=[None, self.output_vector_size])
                    # compute the mean of all predictions
                    accuracy = tf.reduce_sum(tf.pow(pre - lbl, 2)) / (2 * tf.cast(tf.shape(lbl)[0], tf.float32))

                    # accuracy = tf.reduce_mean(tf.cast(tf.nn.sigmoid_cross_entropy_with_logits(logits=pre, labels=lbl), "float"))

        self.log.info('running the session...')

        self.train_pairwise(graph, self.train_dataset, self.train_labels, self.valid_dataset, self.valid_labels, self.test_dataset, self.test_labels,
                       tf_train_left, tf_train_right,
                       tf_train_labels, tf_valid_left, tf_valid_right, tf_test_left, tf_test_right, train_prediction, valid_prediction,
                       test_prediction, loss, optimizer, accuracy, pre, lbl, beta_regu, prob=True)


    @property
    def simple_NN_pairwise(self):
        self.log.info("creating the computational graph...")
        graph = tf.Graph()
        with graph.as_default():
            with tf.device(self.mode):
                # Input data
                tf_train_left = tf.placeholder(tf.int32, shape=(NNConfig.batch_size, self.input_vector_size))
                tf_train_right = tf.placeholder(tf.int32, shape=(NNConfig.batch_size, self.input_vector_size))
                tf_train_labels = tf.placeholder(tf.float32, shape=(NNConfig.batch_size, self.output_vector_size))

                tf_valid_left = tf.placeholder(tf.int32, shape=(NNConfig.batch_size, self.input_vector_size))
                tf_valid_right = tf.placeholder(tf.int32, shape=(NNConfig.batch_size, self.input_vector_size))

                tf_test_left = tf.placeholder(tf.int32, shape=(NNConfig.batch_size, self.input_vector_size))
                tf_test_right = tf.placeholder(tf.int32, shape=(NNConfig.batch_size, self.input_vector_size))

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

                w_h = init_weights([NNConfig.embedding_dim, NNConfig.num_hidden_nodes])
                b_h = init_biases([NNConfig.num_hidden_nodes])

                w_o = init_weights([NNConfig.num_hidden_nodes, self.output_vector_size])
                b_o = init_biases([self.output_vector_size])

                # Embedding layer
                with tf.device('/cpu:0'), tf.name_scope("embedding"):
                    self.W = tf.Variable(
                        tf.random_uniform([self.d_loader.d_handler.get_vocab_size(), NNConfig.embedding_dim], -1.0,
                                          1.0),
                        name="W")
                    embedded_left = tf.nn.embedding_lookup(self.W, tf_train_left)
                    self.embedded_train_left = tf.reduce_sum(embedded_left, [1])

                    embedded_right = tf.nn.embedding_lookup(self.W, tf_train_right)
                    self.embedded_train_right = tf.reduce_sum(embedded_right, [1])

                    embedded_valid = tf.nn.embedding_lookup(self.W, tf_valid_left)
                    self.embedded_valid_left = tf.reduce_sum(embedded_valid, [1])

                    embedded_valid = tf.nn.embedding_lookup(self.W, tf_valid_right)
                    self.embedded_valid_right = tf.reduce_sum(embedded_valid, [1])

                    embedded_test = tf.nn.embedding_lookup(self.W, tf_test_left)
                    self.embedded_test_left = tf.reduce_sum(embedded_test, [1])

                    embedded_test = tf.nn.embedding_lookup(self.W, tf_test_right)
                    self.embedded_test_right = tf.reduce_sum(embedded_test, [1])



                # Training computation
                def model(dataset, w_h, b_h, w_o, b_o, train_mode):
                    if NNConfig.dropout and train_mode:
                        drop_h = dataset
                        for i in range(0, NNConfig.num_hidden_layers):
                            drop_i = tf.nn.dropout(drop_h, NNConfig.dropout_keep_prob_input)
                            w_h = init_weights([drop_h.shape.as_list()[1], NNConfig.num_hidden_nodes])
                            b_h = init_biases([NNConfig.num_hidden_nodes])

                            h_lay_train = tf.nn.relu(tf.matmul(drop_i, w_h) + b_h)
                            drop_h = tf.nn.dropout(h_lay_train, NNConfig.dropout_keep_prob_hidden)

                        return tf.matmul(drop_h, w_o) + b_o
                    else:
                        h_lay_train = dataset
                        for i in range(0, NNConfig.num_hidden_layers):
                            w_h = init_weights([h_lay_train.shape.as_list()[1], NNConfig.num_hidden_nodes])
                            b_h = init_biases([NNConfig.num_hidden_nodes])
                            h_lay_train = tf.nn.relu(tf.matmul(h_lay_train, w_h) + b_h)  # or tf.nn.sigmoid

                        return tf.matmul(h_lay_train, w_o) + b_o

                def pairwise_model(data_left, data_right, w_h, b_h, w_o, b_o, train_mode):
                    logits_left = model(data_left, w_h, b_h, w_o, b_o, train_mode)
                    logits_right = model(data_right, w_h, b_h, w_o, b_o, train_mode)

                    logits = logits_left - logits_right
                    return logits

                logits = pairwise_model(self.embedded_train_left, self.embedded_train_right, w_h, b_h, w_o, b_o, True)

                loss = tf.losses.hinge_loss(labels=tf_train_labels, logits=logits)

                self.log.info("embedded_train shape: {}".format(tf.shape(self.embedded_train_left)))

                # loss = tf.reduce_sum(tf.pow(logits - tf_train_labels, 2)) / (2 * NNConfig.batch_size)
                # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
                # tf.nn.sigmoid_cross_entropy_with_logits instead of tf.nn.softmax_cross_entropy_with_logits for multi-label case
                if NNConfig.regularization:
                    loss += beta_regu * (tf.nn.l2_loss(w_h) + tf.nn.l2_loss(w_o))
                if NNConfig.learning_rate_decay:
                    learning_rate = tf.train.exponential_decay(NNConfig.learning_rate, global_step,
                                                               NNConfig.decay_steps, NNConfig.decay_rate,
                                                               staircase=True)
                    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
                else:
                    optimizer = tf.train.AdamOptimizer(NNConfig.learning_rate).minimize(loss)
                    # optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(loss)

                # score model: linear activation
                train_prediction = logits
                valid_prediction = pairwise_model(self.embedded_valid_left, self.embedded_valid_right, w_h, b_h, w_o, b_o, False)
                test_prediction = pairwise_model(self.embedded_test_left, self.embedded_test_right, w_h, b_h, w_o, b_o, False)

                '''
                run accuracy scope
                '''
                with tf.name_scope('accuracy'):
                    pre = tf.placeholder("float", shape=[None, self.output_vector_size])
                    lbl = tf.placeholder("float", shape=[None, self.output_vector_size])
                    # compute the mean of all predictions
                    accuracy = tf.reduce_sum(tf.pow(pre - lbl, 2)) / (2 * tf.cast(tf.shape(lbl)[0], tf.float32))

                    # accuracy = tf.reduce_mean(tf.cast(tf.nn.sigmoid_cross_entropy_with_logits(logits=pre, labels=lbl), "float"))

        self.log.info('running the session...')

        self.train_pairwise(graph, self.train_dataset, self.train_labels, self.valid_dataset, self.valid_labels, self.test_dataset, self.test_labels,
                       tf_train_left, tf_train_right,
                       tf_train_labels, tf_valid_left, tf_valid_right, tf_test_left, tf_test_right, train_prediction, valid_prediction,
                       test_prediction, loss, optimizer, accuracy, pre, lbl, beta_regu)


    def simple_NN(self):
        self.log.info("creating the computational graph...")
        graph = tf.Graph()
        with graph.as_default():
            with tf.device(self.mode):
                # Input data
                tf_train_dataset = tf.placeholder(tf.int32, shape=(NNConfig.batch_size, self.input_vector_size))
                tf_train_labels = tf.placeholder(tf.float32, shape=(NNConfig.batch_size, self.output_vector_size))

                # Do not load data to constant!
                # tf_valid_dataset = tf.constant(self.valid_dataset)
                # tf_test_dataset = tf.constant(self.test_dataset)

                # create a placeholder
                tf_valid_dataset = tf.placeholder(tf.int32, shape=(NNConfig.batch_size, self.input_vector_size))
                #tf_valid_dataset = tf.Variable(tf_valid_dataset_init)

                tf_test_dataset = tf.placeholder(tf.int32, shape=(NNConfig.batch_size, self.input_vector_size))
                #tf_test_dataset = tf.Variable(tf_test_dataset_init)

                if NNConfig.regularization:
                    beta_regu = tf.placeholder(tf.float32)

                if NNConfig.learning_rate_decay:
                    global_step = tf.Variable(0)

                # Variables.
                def init_weights(shape):
                    # return tf.Variable(tf.random_normal(shape, stddev=0.01))
                    return tf.Variable(tf.truncated_normal(shape))


                def init_biases(shape):
                    return  tf.Variable(tf.zeros(shape))

                w_h = init_weights([NNConfig.embedding_dim, NNConfig.num_hidden_nodes])
                b_h = init_biases([NNConfig.num_hidden_nodes])

                w_o = init_weights([NNConfig.num_hidden_nodes, self.output_vector_size])
                b_o = init_biases([self.output_vector_size])

                # Embedding layer
                with tf.device('/cpu:0'), tf.name_scope("embedding"):
                    self.W = tf.Variable(
                        tf.random_uniform([self.d_loader.d_handler.get_vocab_size(), NNConfig.embedding_dim], -1.0, 1.0),
                        name="W")
                    embedded_train = tf.nn.embedding_lookup(self.W, tf_train_dataset)
                    self.embedded_train_expanded = tf.reduce_sum(embedded_train, [1])

                    embedded_valid = tf.nn.embedding_lookup(self.W, tf_valid_dataset)
                    self.embedded_valid_expanded = tf.reduce_sum(embedded_valid, [1])

                    embedded_test = tf.nn.embedding_lookup(self.W, tf_test_dataset)
                    self.embedded_test_expanded = tf.reduce_sum(embedded_test, [1])



                # Training computation
                def model(dataset, w_h, b_h, w_o, b_o, train):
                    if NNConfig.dropout and train:
                        drop_h = dataset
                        for i in range(0, NNConfig.num_hidden_layers):
                            drop_i = tf.nn.dropout(drop_h, NNConfig.dropout_keep_prob_input)
                            w_h = init_weights([drop_h.shape.as_list()[1], NNConfig.num_hidden_nodes])
                            b_h = init_biases([NNConfig.num_hidden_nodes])

                            h_lay_train = tf.nn.relu(tf.matmul(drop_i, w_h) + b_h)
                            drop_h = tf.nn.dropout(h_lay_train, NNConfig.dropout_keep_prob_hidden)

                        return tf.matmul(drop_h, w_o) + b_o
                    else:
                        h_lay_train = dataset
                        for i in range(0, NNConfig.num_hidden_layers):
                            w_h = init_weights([h_lay_train.shape.as_list()[1], NNConfig.num_hidden_nodes])
                            b_h = init_biases([NNConfig.num_hidden_nodes])
                            h_lay_train = tf.nn.relu(tf.matmul(h_lay_train, w_h) + b_h)  # or tf.nn.sigmoid

                        return tf.matmul(h_lay_train, w_o) + b_o


                logger.info("embedded_train shape: {}".format(tf.shape(self.embedded_train_expanded)))


                logits = model(self.embedded_train_expanded, w_h, b_h, w_o, b_o, True)
                loss = tf.reduce_sum(tf.pow(logits - tf_train_labels, 2)) / (2 * NNConfig.batch_size)

                #loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
                # tf.nn.sigmoid_cross_entropy_with_logits instead of tf.nn.softmax_cross_entropy_with_logits for multi-label case
                if NNConfig.regularization:
                    loss += beta_regu * (tf.nn.l2_loss(w_h) + tf.nn.l2_loss(w_o))
                if NNConfig.learning_rate_decay:
                    learning_rate = tf.train.exponential_decay(NNConfig.learning_rate,global_step,
                                                               NNConfig.decay_steps, NNConfig.decay_rate, staircase=True)
                    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step = global_step)
                else:
                    optimizer = tf.train.AdamOptimizer(NNConfig.learning_rate).minimize(loss)
                    # optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(loss)

                # score model: linear activation
                train_prediction = logits
                valid_prediction = model(self.embedded_valid_expanded, w_h, b_h, w_o, b_o, False)
                test_prediction = model(self.embedded_test_expanded, w_h, b_h, w_o, b_o, False)

                '''
                run accuracy scope
                '''
                with tf.name_scope('accuracy'):
                    pre = tf.placeholder("float", shape=[None, self.output_vector_size])
                    lbl = tf.placeholder("float", shape=[None, self.output_vector_size])
                    # compute the mean of all predictions
                    accuracy = tf.reduce_sum(tf.pow(pre - lbl, 2)) / (2 * tf.cast(tf.shape(lbl)[0], tf.float32))

                    # accuracy = tf.reduce_mean(tf.cast(tf.nn.sigmoid_cross_entropy_with_logits(logits=pre, labels=lbl), "float"))

        self.log.info('running the session...')
        self.train( graph, tf_train_dataset, tf_train_labels,
              tf_valid_dataset, tf_test_dataset, train_prediction, valid_prediction,
              test_prediction, loss, optimizer, accuracy, pre, lbl, beta_regu)


if __name__ == '__main__':
    parser = ArgumentParser(description='Required arguments')
    parser.add_argument('-m', '--mode', help='computation mode', required=False)
    parser.add_argument('-s', '--query_size', help='number of queries for training', required=False)
    parser.add_argument('-l', '--loss', help='loss function [point-wise, pair-wise]', required=False)
    args = parser.parse_args()
    nn = NN(qsize=args.query_size)
    try:
        if args.mode == "gpu": nn.mode = "/gpu:0"
        else: nn.mode = "/cpu:0"
        if args.loss == "pointwise":
            nn.log.info("learn with point-wise loss")
            nn.simple_NN()
        elif args.loss == "pairwise":
            nn.log.info("learn with pair-wise")
            nn.simple_NN_pairwise
        else:
            nn.log.info("learn with cross entropy")
            nn.simple_NN_prob()
        nn.log.info("done..")
    except Exception as e:
        nn.log.exception(e)
        raise











