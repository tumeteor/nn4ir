import numpy as np
import tensorflow as tf
import logging
import sys
import os
import math
sys.path.insert(0, os.path.abspath('..'))

from Util.dataloader import DataLoader
from Util.configs import NNConfig


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

class NN(object):

    def __init__(self):
        self.d_loader = DataLoader()
        self.input_vector_size = self.d_loader.d_handler.get_vocab_size()
        self.output_vector_size = self.d_loader.d_handler.get_vocab_size()
        self.train_dataset, self.train_labels, self.valid_dataset, \
        self.valid_labels, self.test_dataset, self.test_labels = self.d_loader.get_ttv()


    def simple_NN(self):
        logger.info("creating the computational graph...")
        graph = tf.Graph()
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

                # Variables.
                def init_weights(shape):
                    # return tf.Variable(tf.random_normal(shape, stddev=0.01))
                    return tf.Variable(tf.truncated_normal(shape))


                def init_biases(shape):
                    return  tf.Variable(tf.zeros(shape))

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
                        return  tf.matmul(drop_h, w_o) + b_o
                    else:
                        h_lay_train = tf.nn.relu(tf.matmul(dataset, w_h) + b_h)  # or tf.nn.sigmoid
                        return tf.matmul(h_lay_train, w_o) + b_o


                logits = model(tf_train_dataset, w_h, b_h, w_o, b_o, True)
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
                # tf.nn.sigmoid_cross_entropy_with_logits instead of tf.nn.softmax_cross_entropy_with_logits for multi-label case
                if NNConfig.regularization:
                    loss += beta_regu * (tf.nn.l2_loss(w_h) + tf.nn.l2_loss(w_o))
                if NNConfig.learning_rate_decay:
                    learning_rate = tf.train.exponential_decay(NNConfig.learning_rate,global_step,
                                                               NNConfig.decay_steps, NNConfig.decay_rate, staircase=True)
                    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = global_step)
                else:
                    optimizer = tf.train.GradientDescentOptimizer(NNConfig.learning_rate).minimize(loss)
                    # optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(loss)


                train_prediction = tf.nn.softmax(logits)
                valid_prediction = tf.nn.softmax(model(tf_valid_dataset, w_h, b_h, w_o, b_o, False))
                test_prediction = tf.nn.softmax(model(tf_test_dataset, w_h, b_h, w_o, b_o, False))

                with tf.name_scope('accuracy'):
                    pre = tf.placeholder("float", shape=[None, self.output_vector_size])
                    lbl = tf.placeholder("float", shape=[None, self.output_vector_size])
                    accuracy = tf.reduce_mean(tf.cast(tf.nn.sigmoid_cross_entropy_with_logits(logits=pre, labels=lbl), "float"))

        logger.info('running the session...')
        with tf.Session(graph=graph,config=tf.ConfigProto(log_device_placement=True)) as session:
            session.run(tf.global_variables_initializer(), feed_dict={tf_valid_dataset_init: self.valid_dataset,
                                                                      tf_test_dataset_init: self.test_dataset})
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
                                                                           feed_dict={pre: predictions, lbl: batch_labels}))
                    # self.print_words(predictions, batch_labels)
                    logger.info('Validation accuracy:  %.3f%%' % session.run(accuracy,
                                                                             feed_dict={pre: valid_prediction.eval(),
                                                                                        lbl: self.valid_labels}))
                    logger.info('Test accuracy:  %.3f%%' % session.run(accuracy,
                                                                       feed_dict={pre: test_prediction.eval(), lbl: self.test_labels}))
            self.print_words(test_prediction.eval(), self.test_labels)


    def simple_NN_w_CandidateSampling(self):
        print("creating the computational graph...")
        graph = tf.Graph()
        with graph.as_default():
            with tf.device("/cpu:0"):
                # Input data
                tf_train_dataset = tf.placeholder(tf.float32, shape=(NNConfig.batch_size, self.input_vector_size))
                tf_train_labels = tf.placeholder(tf.float32, shape=(NNConfig.batch_size, self.output_vector_size))
                tf_valid_dataset = tf.constant(self.valid_dataset)
                tf_test_dataset = tf.constant(self.test_dataset)
                if NNConfig.regularization:
                    beta_regu = tf.placeholder(tf.float32)
                if NNConfig.learning_rate_decay:
                    global_step = tf.Variable(0)

                # Variables.
                # def init_weights(shape):
                #   # return tf.Variable(tf.random_normal(shape, stddev=0.01))
                #   return tf.Variable(tf.truncated_normal(shape))

                def init_biases(shape):
                    return  tf.Variable(tf.zeros(shape))

                w_h =  tf.Variable(tf.truncated_normal([self.input_vector_size, NNConfig.num_hidden_nodes],
                                                       stddev=1.0 / math.sqrt(NNConfig.num_hidden_nodes)))
                b_h = init_biases([NNConfig.num_hidden_nodes])

                w_o =  tf.Variable(tf.truncated_normal([NNConfig.num_hidden_nodes, self.output_vector_size],
                                                       stddev=1.0 / math.sqrt(self.output_vector_size)))
                b_o = init_biases([self.output_vector_size])

                # Training computation
                def model(dataset, w_h, b_h, w_o, b_o, train):
                    if NNConfig.dropout and train:
                        drop_i = tf.nn.dropout(dataset, NNConfig.dropout_keep_prob_input)
                        h_lay_train = tf.nn.relu(tf.matmul(drop_i, w_h) + b_h)
                        drop_h = tf.nn.dropout(h_lay_train, NNConfig.dropout_keep_prob_hidden)
                        return  tf.matmul(drop_h, w_o) + b_o
                    else:
                        h_lay_train = tf.nn.relu(tf.matmul(dataset, w_h) + b_h)  # or tf.nn.sigmoid
                        return tf.matmul(h_lay_train, w_o) + b_o


                logits = model(tf_train_dataset, w_h, b_h, w_o, b_o, True)

                # sampled_values =


                if NNConfig.candidate_sampling == 'nce_loss': # Noise-contrastive estimation:
                    instances_loss = tf.nn.nce_loss(w_o, b_o, tf_train_dataset, tf_train_labels,
                                                    NNConfig.num_sampled, self.output_vector_size, num_true=10)
                elif NNConfig.candidate_sampling == 'softmax_loss':
                    instances_loss = tf.nn.sampled_softmax_loss(w_o, b_o, tf_train_dataset, tf_train_labels,
                                                                NNConfig.num_sampled, self.output_vector_size, num_true=10)
                else:
                    instances_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf_train_labels)
                    print('no candidate sampling....')
                loss = tf.reduce_mean(instances_loss)

                if NNConfig.regularization:
                    loss += beta_regu * (tf.nn.l2_loss(w_h) + tf.nn.l2_loss(w_o))
                if NNConfig.learning_rate_decay:
                    learning_rate = tf.train.exponential_decay(NNConfig.learning_rate,global_step,
                                                               NNConfig.decay_steps, NNConfig.decay_rate, staircase=True)
                    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = global_step)
                else:
                    optimizer = tf.train.GradientDescentOptimizer(NNConfig.learning_rate).minimize(loss)
                    # optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(loss)


                train_prediction = tf.nn.softmax(logits)
                valid_prediction = tf.nn.softmax(model(tf_valid_dataset, w_h, b_h, w_o, b_o, False))
                test_prediction = tf.nn.softmax(model(tf_test_dataset, w_h, b_h, w_o, b_o, False))

                with tf.name_scope('accuracy'):
                    pre = tf.placeholder("float", shape=[None, self.output_vector_size])
                    lbl = tf.placeholder("float", shape=[None, self.output_vector_size])
                    accuracy = tf.reduce_mean(tf.cast(tf.nn.sigmoid_cross_entropy_with_logits(logits=pre, labels=lbl), "float"))

        logger.info('running the session...')
        with tf.Session(graph=graph,config=tf.ConfigProto(log_device_placement=True)) as session:
            tf.initialize_all_variables().run()
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
                                                                           feed_dict={pre: predictions, lbl: batch_labels}))
                    # self.print_words(predictions, batch_labels)
                    logger.info('Validation accuracy:  %.3f%%' % session.run(accuracy,
                                                                             feed_dict={pre: valid_prediction.eval(),
                                                                                        lbl: self.valid_labels}))
            logger.info('Test accuracy:  %.3f%%' % session.run(accuracy,
                                                               feed_dict={pre: test_prediction.eval(), lbl: self.test_labels}))
            self.print_words(test_prediction.eval(), self.test_labels)





    def simple_NN_w_embedding(self):
        logger.info("creating the computational graph...")
        graph = tf.Graph()
        #sess = tf.Session(graph=graph)
        with graph.as_default():
            with tf.device("/cpu:0"):
                # Input data
                tf_train_dataset = tf.placeholder(tf.float32, shape=(NNConfig.batch_size, self.input_vector_size))
                tf_train_labels = tf.placeholder(tf.float32, shape=(NNConfig.batch_size, self.output_vector_size))

                # Do not load data to constant!
                #tf_valid_dataset = tf.constant(self.valid_dataset)
                #tf_test_dataset = tf.constant(self.test_dataset)

                # create a placeholder
                tf_valid_dataset_init = tf.placeholder(tf.float32, shape=self.valid_dataset.shape)
                tf_valid_dataset = tf.Variable(tf_valid_dataset_init)

                tf_test_dataset_init = tf.placeholder(tf.float32, shape=self.test_dataset.shape)
                tf_test_dataset = tf.Variable(tf_test_dataset_init)

                if NNConfig.regularization:
                    beta_regu = tf.placeholder(tf.float32)

                if NNConfig.learning_rate_decay:
                    global_step = tf.Variable(0)

                # Look up embeddings for inputs.
                #train_embeddings = tf.Variable(
                #   tf.random_uniform([self.input_vector_size, NNConfig.embedding_size], -1.0, 1.0))
                #train_embed = tf.nn.embedding_lookup(train_embeddings, tf_train_dataset)

                #valid_embeddings = tf.Variable(
                #    tf.random_uniform([self.input_vector_size, NNConfig.embedding_size], -1.0, 1.0))
                #valid_embed = tf.nn.embedding_lookup(train_embeddings, tf_valid_dataset)



                # Variables.
                def init_weights(shape):
                    # return tf.Variable(tf.random_normal(shape, stddev=0.01))
                    return tf.Variable(tf.truncated_normal(shape))


                def init_biases(shape):
                    return  tf.Variable(tf.zeros(shape))

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
                        return  tf.matmul(drop_h, w_o) + b_o
                    else:
                        h_lay_train = tf.nn.relu(tf.matmul(dataset, w_h) + b_h)  # or tf.nn.sigmoid
                        return tf.matmul(h_lay_train, w_o) + b_o


                logits = model(tf_train_dataset, w_h, b_h, w_o, b_o, True)
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
                # tf.nn.sigmoid_cross_entropy_with_logits instead of tf.nn.softmax_cross_entropy_with_logits for multi-label case
                if NNConfig.regularization:
                    loss += beta_regu * (tf.nn.l2_loss(w_h) + tf.nn.l2_loss(w_o))
                if NNConfig.learning_rate_decay:
                    learning_rate = tf.train.exponential_decay(NNConfig.learning_rate,global_step,
                                                               NNConfig.decay_steps, NNConfig.decay_rate, staircase=True)
                    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = global_step)
                else:
                    optimizer = tf.train.GradientDescentOptimizer(NNConfig.learning_rate).minimize(loss)
                    # optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(loss)


                train_prediction = tf.nn.softmax(logits)
                valid_prediction = tf.nn.softmax(model(tf_valid_dataset, w_h, b_h, w_o, b_o, False))
                test_prediction = tf.nn.softmax(model(tf_test_dataset, w_h, b_h, w_o, b_o, False))

                with tf.name_scope('accuracy'):
                    pre = tf.placeholder("float", shape=[None, self.output_vector_size])
                    lbl = tf.placeholder("float", shape=[None, self.output_vector_size])
                    accuracy = tf.reduce_mean(tf.cast(tf.nn.sigmoid_cross_entropy_with_logits(logits=pre, labels=lbl), "float"))

        logger.info('running the session...')
        with tf.Session(graph=graph,config=tf.ConfigProto(log_device_placement=True)) as session:
            session.run(tf.global_variables_initializer(), feed_dict={tf_valid_dataset_init: self.valid_dataset,
                                                                      tf_test_dataset_init: self.test_dataset})
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
                                                                           feed_dict={pre: predictions, lbl: batch_labels}))
                    # self.print_words(predictions, batch_labels)
                    logger.info('Validation accuracy:  %.3f%%' % session.run(accuracy,
                                                                             feed_dict={pre: valid_prediction.eval(),
                                                                                        lbl: self.valid_labels}))
                    logger.info('Test accuracy:  %.3f%%' % session.run(accuracy,
                                                                       feed_dict={pre: test_prediction.eval(), lbl: self.test_labels}))
            self.print_words(test_prediction.eval(), self.test_labels)



    def print_words(self, preds, labels):
        for pred, label in zip(preds,labels):
            label_ids = self.d_loader.d_handler.get_ids_from_binary_vector(label)[0]
            pred_ids = np.argsort(np.negative(pred))[:label_ids.size]
            # pred_ids = np.argsort(pred)[(-(label_ids.size)):][::-1]
            # print(label_ids)
            # print(pred_ids)
            print(self.d_loader.d_handler.id_list_to_word_list(label_ids),"-->" ,self.d_loader.d_handler.id_list_to_word_list(pred_ids))
            # break

    def get_words(self,vect):
        ids = self.d_loader.d_handler.get_ids_from_binary_vector(vect)[0]
        return self.d_loader.d_handler.id_list_to_word_list(ids)


if __name__ == '__main__':
    try:
        nn = NN()
        nn.simple_NN()
        logger.info("done...")
    except Exception as e:
        logger.exception(e)
        raise