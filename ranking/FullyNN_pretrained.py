import numpy as np
import tensorflow as tf
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from argparse import ArgumentParser
from ranking.NN import NN
from Util.dataloader import DataLoader
from Util.configs import NNConfig

class NN(NN):
    def __init__(self, qsize="1k"):
        super(NN, self).__init__()
        self.input_vector_size = NNConfig.max_doc_size
        # output vector size = 1 for scoring model
        self.output_vector_size = 1
        self.d_loader = DataLoader(pretrained=True, qsize=qsize)
        self.train_dataset, self.train_labels, self.valid_dataset, \
        self.valid_labels, self.test_dataset, self.test_labels = self.d_loader.get_ttv()
        self.embedding = None
        self.embedding_init = None
        self.embedding_placeholder = None
        self.pretrained = True

    def simple_NN(self, mode="/cpu:0"):
        self.log.info("creating the computational graph...")
        graph = tf.Graph()
        # sess = tf.Session(graph=graph)
        with graph.as_default():
            with tf.device("/cpu:0"):
                tf_train_dataset = tf.placeholder(tf.int32, shape=(NNConfig.batch_size, self.input_vector_size))
                tf_train_labels = tf.placeholder(tf.float32, shape=(NNConfig.batch_size, self.output_vector_size))

                # Do not load data to constant!
                # tf_valid_dataset = tf.constant(self.valid_dataset)
                # tf_test_dataset = tf.constant(self.test_dataset)

                # create a placeholder
                tf_valid_dataset = tf.placeholder(tf.int32, shape=(NNConfig.batch_size, self.input_vector_size))
                # tf_valid_dataset = tf.Variable(tf_valid_dataset_init)

                tf_test_dataset = tf.placeholder(tf.int32, shape=(NNConfig.batch_size, self.input_vector_size))
                # tf_test_dataset = tf.Variable(tf_test_dataset_init)

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

                # w_h = init_weights([NNConfig.embedding_dim, NNConfig.num_hidden_nodes])
                # b_h = init_biases([NNConfig.num_hidden_nodes])

                w_o = init_weights([NNConfig.num_hidden_nodes, self.output_vector_size])
                b_o = init_biases([self.output_vector_size])

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
                self.embedded_valid_expanded = tf.reduce_sum(tf.nn.embedding_lookup(W, tf_valid_dataset), [1])
                self.embedded_test_expanded = tf.reduce_sum(tf.nn.embedding_lookup(W, tf_test_dataset), [1])


            def model(dataset, w_o, b_o, train_mode):
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

            self.log.info("embedded_train shape: {}".format(tf.shape(self.embedded_train_expanded)))

            logits, w_hs = model(self.embedded_train_expanded, w_o, b_o, True)
            loss = tf.reduce_sum(tf.pow(logits - tf_train_labels, 2)) / (2 * NNConfig.batch_size)

            # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
            # tf.nn.sigmoid_cross_entropy_with_logits instead of tf.nn.softmax_cross_entropy_with_logits for multi-label case
            if NNConfig.regularization:
                loss += beta_regu * (sum(tf.nn.l2_loss(w_h) for w_h in w_hs) + tf.nn.l2_loss(w_o))
            if NNConfig.learning_rate_decay:
                learning_rate = tf.train.exponential_decay(NNConfig.learning_rate, global_step,
                                                           NNConfig.decay_steps, NNConfig.decay_rate, staircase=True)
                optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
            else:
                optimizer = tf.train.AdamOptimizer(NNConfig.learning_rate).minimize(loss)
                # optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(loss)

            # score model: linear activation
            train_prediction = logits
            valid_prediction, w_v = model(self.embedded_valid_expanded, w_o, b_o, False)
            test_prediction, w_t = model(self.embedded_test_expanded, w_o, b_o, False)

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
        self.train(graph, tf_train_dataset, tf_train_labels,
                   tf_valid_dataset, tf_test_dataset, train_prediction, valid_prediction,
                   test_prediction, loss, optimizer, accuracy, pre, lbl, beta_regu)


if __name__ == '__main__':
    parser = ArgumentParser(description='Required arguments')
    parser.add_argument('-m', '--mode', help='computation mode', required=False)
    parser.add_argument('-s', '--query_size', help='number of queries for training', required=False)
    parser.add_argument('-l', '--loss', help='loss function [point-wise, pair-wise]', required=False)
    args = parser.parse_args()
    nn = NN() if args.query_size is None else NN(args.query_size)
    try:
        if args.mode == "gpu": nn.mode = "/gpu:0"
        else: nn.mode = "/cpu:0"
        if args.loss == "point-wise" or args.loss == "pair-wise":
            nn.log.info("learn with pair-wise loss")
            nn.lf = args.loss
        else: nn.lf == "point-wise"
        nn.simple_NN()
        nn.log.info("done..")
    except Exception as e:
        nn.log.exception(e)
        raise