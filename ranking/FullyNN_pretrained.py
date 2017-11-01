import numpy as np
import tensorflow as tf
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from argparse import ArgumentParser
from ranking.NN import NN
from Util.dataloader import DataLoader
from Util.configs import NNConfig, DataConfig

import logging
logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
fileHandler = logging.FileHandler("{0}/{1}.log".format("./", "FullyNN_pretrained"))
fileHandler.setFormatter(logFormatter)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(fileHandler)
logger.addHandler(consoleHandler)

class NN(NN):
    def __init__(self):
        self.d_loader = DataLoader(pretrained=True)
        self.train_dataset, self.train_labels, self.valid_dataset, \
        self.valid_labels, self.test_dataset, self.test_labels = self.d_loader.get_ttv()

    def simple_NN_w_embedding(self, mode="/cpu:0"):
        logger.info("creating the computational graph...")
        graph = tf.Graph()
        # sess = tf.Session(graph=graph)
        with graph.as_default():
            with tf.device("/cpu:0"):
                # Input data
                tf_train_dataset = tf.placeholder(tf.int32, shape=(NNConfig.batch_size, DataConfig.max_doc_size))
                tf_train_labels = tf.placeholder(tf.float32, shape=(NNConfig.batch_size, 1))

                # Do not load data to constant!
                # tf_valid_dataset = tf.constant(self.valid_dataset)
                # tf_test_dataset = tf.constant(self.test_dataset)

                # create a placeholder
                tf_valid_dataset_init = tf.placeholder(tf.int32, shape=self.valid_dataset.shape)
                tf_valid_dataset = tf.Variable(tf_valid_dataset_init)

                tf_test_dataset_init = tf.placeholder(tf.int32, shape=self.test_dataset.shape)
                tf_test_dataset = tf.Variable(tf_test_dataset_init)

                if NNConfig.regularization:
                    beta_regu = tf.placeholder(tf.float32)

                if NNConfig.learning_rate_decay:
                    global_step = tf.Variable(0)


                embed_vocab_size = len(self.d_loader.pretrain_vocab)
                embedding_dim = len(self.d_loader.embd[0])
                embedding = np.asarray(self.d_loader.embd)

                W = tf.Variable(tf.constant(0.0, shape=[embed_vocab_size, embedding_dim]),
                                trainable=False, name="W")

                embedding_placeholder = tf.placeholder(tf.float32, [embed_vocab_size, NNConfig.embedding_dim])
                embedding_init = W.assign(embedding_placeholder)

                # Reduce along dimension 1 (`n_input`) to get a single vector (row)
                # per input example. It's fairly typical to do this for bag-of-words type problems.

                train_embed = tf.reduce_sum(tf.nn.embedding_lookup(W, tf_train_dataset), [1])
                valid_embed = tf.reduce_sum(tf.nn.embedding_lookup(W, tf_valid_dataset), [1])
                test_embed = tf.reduce_sum(tf.nn.embedding_lookup(W, tf_test_dataset), [1])

                # Variables.
                def init_weights(shape):
                    # return tf.Variable(tf.random_normal(shape, stddev=0.01))
                    return tf.Variable(tf.truncated_normal(shape))

                def init_biases(shape):
                    return tf.Variable(tf.zeros(shape))

                w_h = init_weights([embedding_dim, NNConfig.num_hidden_nodes])
                b_h = init_biases([NNConfig.num_hidden_nodes])

                w_o = init_weights([NNConfig.num_hidden_nodes, 1])
                b_o = init_biases([1])

                # Training computation
                def model(dataset, w_h, b_h, w_o, b_o, train):
                    if NNConfig.dropout and train:
                        drop_i = tf.nn.dropout(dataset, NNConfig.dropout_keep_prob_input)
                        h_lay_train = tf.nn.relu(tf.matmul(drop_i, w_h) + b_h)
                        drop_h = tf.nn.dropout(h_lay_train, NNConfig.dropout_keep_prob_hidden)
                        return tf.matmul(drop_h, w_o) + b_o
                    else:
                        h_lay_train = tf.nn.relu(tf.matmul(dataset, w_h) + b_h)  # or tf.nn.sigmoid
                        return tf.matmul(h_lay_train, w_o) + b_o

                logits = model(train_embed, w_h, b_h, w_o, b_o, True)
                loss = tf.reduce_sum(tf.pow(logits - tf_train_labels, 2)) / (2 * (2 * NNConfig.batch_size))
                #loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
                # tf.nn.sigmoid_cross_entropy_with_logits instead of tf.nn.softmax_cross_entropy_with_logits for multi-label case
                if NNConfig.regularization:
                    loss += beta_regu * (tf.nn.l2_loss(w_h) + tf.nn.l2_loss(w_o))
                if NNConfig.learning_rate_decay:
                    learning_rate = tf.train.exponential_decay(NNConfig.learning_rate, global_step,
                                                               NNConfig.decay_steps, NNConfig.decay_rate,
                                                               staircase=True)
                    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
                else:
                    optimizer = tf.train.GradientDescentOptimizer(NNConfig.learning_rate).minimize(loss)
                    # optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(loss)

                #train_prediction = tf.nn.softmax(logits)
                #valid_prediction = tf.nn.softmax(model(valid_embed, w_h, b_h, w_o, b_o, False))
                #test_prediction = tf.nn.softmax(model(test_embed, w_h, b_h, w_o, b_o, False))

                # linear activation
                train_prediction = logits
                valid_prediction = model(valid_embed, w_h, b_h, w_o, b_o, False)
                test_prediction = model(test_embed, w_h, b_h, w_o, b_o, False)

                with tf.name_scope('accuracy'):
                    pre = tf.placeholder("float", shape=[None, 1])
                    lbl = tf.placeholder("float", shape=[None, 1])
                    #accuracy = tf.reduce_mean(
                        #tf.cast(tf.nn.sigmoid_cross_entropy_with_logits(logits=pre, labels=lbl), "float"))
                    accuracy = tf.reduce_sum(tf.pow(pre - lbl, 2)) / (2 * tf.cast(tf.shape(lbl)[0], tf.float32))

        logger.info('running the session...')
        with tf.Session(graph=graph, config=tf.ConfigProto(log_device_placement=True)) as session:
            session.run(tf.global_variables_initializer(), feed_dict={tf_valid_dataset_init: self.valid_dataset,
                                                                      tf_test_dataset_init: self.test_dataset})
            # After creating a session and initialize global variables, run the embedding_init operation by feeding in the 2-D array embedding.
            session.run(embedding_init, feed_dict={embedding_placeholder: embedding})
            logger.info('Initialized')
            for step in range(NNConfig.num_steps):
                offset = (step * NNConfig.batch_size) % (self.train_labels.shape[0] - NNConfig.batch_size)
                batch_data = self.train_dataset[offset:(offset + NNConfig.batch_size), :]
                batch_labels = self.train_labels[offset:(offset + NNConfig.batch_size)]
                batch_labels = batch_labels.reshape(len(batch_labels),1)

                # print('-' * 80)
                # for vec in batch_labels:
                #   print('.' * 200)
                #   print(self.get_words(vec))

                feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
                if NNConfig.regularization:
                    feed_dict[beta_regu] = NNConfig.beta_regu
                logger.debug('start optimizing')
                _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
                logger.debug('optimizing done')
                if (step % NNConfig.summary_steps == 0):
                    logger.info("Minibatch loss at step %d: %f" % (step, l))
                    logger.info("Minibatch accuracy: %.3f%%" % session.run(accuracy,
                                                                           feed_dict={pre: predictions,
                                                                                      lbl: batch_labels}))
                    # self.print_words(predictions, batch_labels)
                    logger.info('Validation accuracy:  %.3f%%' % session.run(accuracy,
                                                                             feed_dict={pre: valid_prediction.eval(),
                                                                                        lbl: self.valid_labels}))
                    logger.info('Test accuracy:  %.3f%%' % session.run(accuracy,
                                                                       feed_dict={pre: test_prediction.eval(),
                                                                                  lbl: self.test_labels}))



if __name__ == '__main__':
    parser = ArgumentParser(description='Required arguments')
    parser.add_argument('-m', '--mode', help='computation mode', required=False)
    args = parser.parse_args()
    nn = NN()
    try:
        if args.mode == "gpu":
            nn.simple_NN(mode="/gpu:0")
        else: nn.simple_NN()
        logger.info("done..")
    except Exception as e:
        logger.exception(e)
        raise
