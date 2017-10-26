import numpy as np
import tensorflow as tf
import logging
import sys
import os
import math
sys.path.insert(0, os.path.abspath('..'))
from tensorflow.contrib import learn

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

class NN(object):
    def __init__(self):
        self.d_loader = DataLoader()
        self.input_vector_size = self.d_loader.d_handler.get_vocab_size()
        self.output_vector_size = self.d_loader.d_handler.get_vocab_size()
        self.train_dataset, self.train_labels, self.valid_dataset, \
        self.valid_labels, self.test_dataset, self.test_labels = self.d_loader.get_ttv()

    def simple_NN_w_embedding(self):
        logger.info("creating the computational graph...")
        graph = tf.Graph()
        # sess = tf.Session(graph=graph)
        with graph.as_default():
            with tf.device("/cpu:0"):
                # Input data
                tf_train_dataset = tf.placeholder(tf.float32, shape=(NNConfig.batch_size, self.input_vector_size))
                tf_train_labels = tf.placeholder(tf.float32, shape=(NNConfig.batch_size, self.output_vector_size))

                # Do not load data to constant!
                # tf_valid_dataset = tf.constant(self.valid_dataset)
                # tf_test_dataset = tf.constant(self.test_dataset)

                # create a placeholder
                tf_valid_dataset_init = tf.placeholder(tf.float32, shape=self.valid_dataset.shape)
                tf_valid_dataset = tf.Variable(tf_valid_dataset_init)

                tf_test_dataset_init = tf.placeholder(tf.float32, shape=self.test_dataset.shape)
                tf_test_dataset = tf.Variable(tf_test_dataset_init)

                if NNConfig.regularization:
                    beta_regu = tf.placeholder(tf.float32)

                if NNConfig.learning_rate_decay:
                    global_step = tf.Variable(0)

                def loadGloVe(glove_file_path):
                    vocab = []
                    embd = []
                    file = open(glove_file_path, 'r')
                    for line in file.readlines():
                        row = line.strip().split('\t')
                        if self.d_loader.d_handler.get_id_of_word(row[1]) != 0:
                            vocab.append()
                        else:
                            continue
                        embd.append(row[2])
                    print('Loaded GloVe!')
                    file.close()
                    return vocab, embd

                '''
                Load pretrained embeddings
                '''
                vocab, embd = loadGloVe(DataConfig.glove_file_path)
                embed_vocab_size = len(vocab)
                embedding_dim = len(embd[0])
                embedding = np.asarray(embd)

                W = tf.Variable(tf.constant(0.0, shape=[embed_vocab_size, embedding_dim]),
                                trainable=False, name="W")

                embedding_placeholder = tf.placeholder(tf.float32, [embed_vocab_size, NNConfig.embedding_dim])
                embedding_init = W.assign(embedding_placeholder)


                train_embed = tf.nn.embedding_lookup(W, tf_train_dataset)
                valid_embed = tf.nn.embedding_lookup(W, tf_valid_dataset)
                test_embed = tf.nn.embedding_lookup(W, tf_test_dataset)

                # Look up embeddings for inputs.
                # train_embeddings = tf.Variable(
                #   tf.random_uniform([self.input_vector_size, NNConfig.embedding_size], -1.0, 1.0))
                # train_embed = tf.nn.embedding_lookup(train_embeddings, tf_train_dataset)

                # valid_embeddings = tf.Variable(
                #    tf.random_uniform([self.input_vector_size, NNConfig.embedding_size], -1.0, 1.0))
                # valid_embed = tf.nn.embedding_lookup(train_embeddings, tf_valid_dataset)



                # Variables.
                def init_weights(shape):
                    # return tf.Variable(tf.random_normal(shape, stddev=0.01))
                    return tf.Variable(tf.truncated_normal(shape))

                def init_biases(shape):
                    return tf.Variable(tf.zeros(shape))

                w_h = init_weights([self.input_vector_size, NNConfig.num_hidden_nodes])
                b_h = init_biases([NNConfig.num_hidden_nodes])

                w_o = init_weights([NNConfig.num_hidden_nodes, self.output_vector_size])
                b_o = init_biases([self.output_vector_size])

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
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
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

                train_prediction = tf.nn.softmax(logits)
                valid_prediction = tf.nn.softmax(model(valid_embed, w_h, b_h, w_o, b_o, False))
                test_prediction = tf.nn.softmax(model(test_embed, w_h, b_h, w_o, b_o, False))

                with tf.name_scope('accuracy'):
                    pre = tf.placeholder("float", shape=[None, self.output_vector_size])
                    lbl = tf.placeholder("float", shape=[None, self.output_vector_size])
                    accuracy = tf.reduce_mean(
                        tf.cast(tf.nn.sigmoid_cross_entropy_with_logits(logits=pre, labels=lbl), "float"))

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
                batch_labels = self.train_labels[offset:(offset + NNConfig.batch_size), :]

                # print('-' * 80)
                # for vec in batch_labels:
                #   print('.' * 200)
                #   print(self.get_words(vec))

                feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
                if NNConfig.regularization:
                    feed_dict[beta_regu] = NNConfig.beta_regu
                _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
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
    try:
        nn = NN()
        nn.simple_NN_w_embedding()
        logger.info("done...")
    except Exception as e:
        logger.exception(e)
        raise