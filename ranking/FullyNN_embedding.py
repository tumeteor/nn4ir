import numpy as np
import tensorflow as tf
from Util.configs import NNConfig, DataConfig
from Util.dataloader import DataLoader
import logging
from argparse import ArgumentParser

import logging
logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
fileHandler = logging.FileHandler("{0}/{1}.log".format("./", "Fullyconnected_NN"))
fileHandler.setFormatter(logFormatter)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(fileHandler)
logger.addHandler(consoleHandler)

class NN:
    def __init__(self):
        self.d_loader = DataLoader(embedding=True)
        self.input_vector_size = DataConfig.max_doc_size
        # output vector size = 1 for scoring model
        self.output_vector_size = 1
        self.train_dataset, self.train_labels, self.valid_dataset, \
        self.valid_labels, self.test_dataset, self.test_labels = self.d_loader.get_ttv()

        self.embedded_train_expanded = None
        self.embedded_valid_expanded = None
        self.embedded_test_expanded = None


    def simple_NN(self, mode="/cpu:0"):
        logger.info("creating the computational graph...")
        graph = tf.Graph()
        with graph.as_default():
            with tf.device(mode):
                # Input data
                tf_train_dataset = tf.placeholder(tf.int32, shape=(NNConfig.batch_size, self.input_vector_size))
                tf_train_labels = tf.placeholder(tf.float32, shape=(NNConfig.batch_size, self.output_vector_size))

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
                        drop_i = tf.nn.dropout(dataset, NNConfig.dropout_keep_prob_input)
                        h_lay_train = tf.nn.relu(tf.matmul(drop_i, w_h) + b_h)
                        drop_h = tf.nn.dropout(h_lay_train, NNConfig.dropout_keep_prob_hidden)
                        return tf.matmul(drop_h, w_o) + b_o
                    else:
                        h_lay_train = tf.nn.relu(tf.matmul(dataset, w_h) + b_h)  # or tf.nn.sigmoid
                        return tf.matmul(h_lay_train, w_o) + b_o

                logger.info("embedded_train shape: {}".format(tf.shape(self.embedded_train_expanded)))
                logits = model(self.embedded_train_expanded, w_h, b_h, w_o, b_o, True)
                loss = tf.reduce_sum(tf.pow(logits - tf_train_labels, 2)) /  \
                       (2 * NNConfig.batch_size)
                #loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
                # tf.nn.sigmoid_cross_entropy_with_logits instead of tf.nn.softmax_cross_entropy_with_logits for multi-label case
                if NNConfig.regularization:
                    loss += beta_regu * (tf.nn.l2_loss(w_h) + tf.nn.l2_loss(w_o))
                if NNConfig.learning_rate_decay:
                    learning_rate = tf.train.exponential_decay(NNConfig.learning_rate,global_step,
                                                               NNConfig.decay_steps, NNConfig.decay_rate, staircase=True)
                    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step = global_step)
                else:
                    optimizer = tf.train.AdamDescentOptimizer(NNConfig.learning_rate).minimize(loss)
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

        logger.info('running the session...')
        with tf.Session(graph=graph,config=tf.ConfigProto(log_device_placement=True)) as session:
            session.run(tf.global_variables_initializer(), feed_dict={tf_valid_dataset_init: self.valid_dataset,
                                                                      tf_test_dataset_init: self.test_dataset})
            logger.info('Initialized')
            for step in range(NNConfig.num_steps):
                offset = (step * NNConfig.batch_size) % (self.train_labels.shape[0] - NNConfig.batch_size)
                batch_data = self.train_dataset[offset:(offset + NNConfig.batch_size), :]
                batch_labels = self.train_labels[offset:(offset + NNConfig.batch_size)]
                batch_labels = batch_labels.reshape(len(batch_labels), 1)

                # print('-' * 80)
                # for vec in batch_labels:
                #   print('.' * 200)
                #   print(self.get_words(vec))

                feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
                if NNConfig.regularization:
                    feed_dict[beta_regu] = NNConfig.beta_regu
                logger.info('start optimizing')
                _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
                logger.info('optimizing finished')
                if (step % NNConfig.summary_steps == 0):
                    logger.info("Minibatch loss at step %d: %f" % (step, l))
                    logger.info("Minibatch accuracy: %.3f%%" % session.run(accuracy,
                                                                           feed_dict={pre: predictions, lbl: batch_labels}))
                    # self.print_words(predictions, batch_labels)
                    logger.info('Validation accuracy:  %.3f%%' % session.run(accuracy,
                                                                             feed_dict={pre: valid_prediction.eval(),
                                                                                        lbl: self.valid_labels}))
                    logger.info('Test accuracy:  %.3f%%' % session.run(accuracy,
                                                                       feed_dict={pre: test_prediction.eval(), lbl: self.test_labels}))
            self.print_words(test_prediction.eval(), self.test_labels)

    def print_words(self, preds, labels):
        for pred, label in zip(preds, labels):
            label_ids = self.d_loader.d_handler.get_ids_from_binary_vector(label)[0]
            pred_ids = np.argsort(np.negative(pred))[:label_ids.size]
            # pred_ids = np.argsort(pred)[(-(label_ids.size)):][::-1]
            # print(label_ids)
            # print(pred_ids)
            print(self.d_loader.d_handler.id_list_to_word_list(label_ids), "-->",
                  self.d_loader.d_handler.id_list_to_word_list(pred_ids))
            # break

    def get_words(self, vect):
        ids = self.d_loader.d_handler.get_ids_from_binary_vector(vect)[0]
        return self.d_loader.d_handler.id_list_to_word_list(ids)


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











