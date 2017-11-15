from abc import abstractmethod
import numpy as np
import itertools
import logging
import tensorflow as tf
from Util.configs import NNConfig
from Util.datautil import Utilities
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
        self.train_dataset = None
        self.train_labels = None
        self.valid_dataset = None
        self.valid_labels = None
        self.test_dataset = None
        self.test_labels = None
        self.log = logger

    @abstractmethod
    def simple_NN(self,mode):
        pass

    def batchData(self, batch_size, data, labels):
        '''
        split data into smaller batch by the first tensor dimension
        :param batch_size:
        :param data: Tensor
        :param labels:
        :return:
        '''
        data_length = len(data)
        steps = int(data_length / batch_size)
        data_batches = list()
        labels_batches = list()
        for step in range(0, steps):
            offset = (step * batch_size) % (data_length - batch_size)
            batch_data = data[offset:(offset + batch_size), :]
            batch_labels = labels[offset:(offset + batch_size)]
            # get rank only from label
            batch_labels = batch_labels[:, 1]
            batch_labels = batch_labels.reshape(batch_size, 1)
            data_batches.append(batch_data)
            labels_batches.append(batch_labels)

        return steps, data_batches, labels_batches

    def batchData_pairwise(self, batch_size, data_left, data_right, labels):
        '''
        split data into smaller batch by the first tensor dimension
        :param batch_size:
        :param data: Tensor
        :param labels:
        :return:
        '''
        assert (len(data_left) == len(data_right))
        data_length = len(data_left)
        steps = int(data_length / batch_size)
        data_left_batches = list()
        data_right_batches = list()
        labels_batches = list()
        for step in range(0, steps):
            offset = (step * batch_size) % (data_length - batch_size)
            batch_data_left = data_left[offset:(offset + batch_size), :]
            batch_data_right = data_left[offset:(offset + batch_size), :]
            batch_labels = labels[offset:(offset + batch_size)]
            # get rank only from label
            batch_labels = batch_labels[:, 1]
            batch_labels = batch_labels.reshape(batch_size, 1)
            data_left_batches.append(batch_data_left)
            data_right_batches.append(batch_data_right)
            labels_batches.append(batch_labels)

        return steps, data_left_batches, data_right_batches, labels_batches

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


    def train(self, graph, tf_train_dataset, tf_train_labels,
              tf_valid_dataset, tf_test_dataset, train_prediction, valid_prediction,
              test_prediction, loss, optimizer, accuracy, pre, lbl, beta_regu):
        with tf.Session(graph=graph, config=tf.ConfigProto(log_device_placement=True)) as session:
            session.run(tf.global_variables_initializer())
            logger.info('Initialized')
            for step in range(NNConfig.num_steps):
                offset = (step * NNConfig.batch_size) % (self.train_labels.shape[0] - NNConfig.batch_size)
                batch_data = self.train_dataset[offset:(offset + NNConfig.batch_size), :]
                batch_labels = self.train_labels[offset:(offset + NNConfig.batch_size)]
                # get rank only from label
                batch_labels = batch_labels[:, 1]
                batch_labels = batch_labels.reshape(len(batch_labels), 1)

                # print('-' * 80)
                # for vec in batch_labels:
                #   print('.' * 200)
                #   print(self.get_words(vec))

                feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
                if NNConfig.regularization:
                    feed_dict[beta_regu] = NNConfig.beta_regu
                logger.debug('start optimizing')
                _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
                logger.debug('optimizing finished')
                if (step % NNConfig.summary_steps == 0):
                    logger.info("Minibatch loss at step %d: %f" % (step, l))
                    logger.info("Minibatch MSE: %.3f" % session.run(accuracy,
                                                                    feed_dict={pre: predictions, lbl: batch_labels}))
                    for i in range(0, 5):
                        print("label value:", batch_labels[i], \
                              "estimated value:", predictions[i])
                    # self.print_words(predictions, batch_labels)
                    steps, valid_data_batches, valid_label_batches = self.batchData(data=self.valid_dataset,
                                                                                    labels=self.valid_labels,
                                                                                    batch_size=NNConfig.batch_size)
                    valid_mse = 0
                    for step in range(0, steps):
                        batch_prediction = session.run(valid_prediction,
                                                       feed_dict={tf_valid_dataset: valid_data_batches[step]})

                        batch_mse = session.run(accuracy, feed_dict={pre: batch_prediction,
                                                                     lbl: valid_label_batches[step]})
                        valid_mse += batch_mse
                    logger.info('Validation MSE: %.3f' % valid_mse)

                    steps, test_data_batches, test_label_batches = self.batchData(data=self.test_dataset,
                                                                                  labels=self.test_labels,
                                                                                  batch_size=NNConfig.batch_size)
                    test_mse = 0
                    for step in range(0, steps):
                        batch_prediction = session.run(test_prediction,
                                                       feed_dict={tf_test_dataset: test_data_batches[step],
                                                                  lbl: test_label_batches[step]})
                        batch_mse = session.run(accuracy, feed_dict={pre: batch_prediction,
                                                                     lbl: test_label_batches[step]})
                        test_mse += batch_mse
                    logger.info('Test MSE: %.3f' % test_mse)

    def train_pairwise(self, graph, train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels,
                       tf_train_left, tf_train_right,
                       tf_train_labels, tf_valid_left, tf_valid_right, tf_test_left, tf_test_right, train_prediction, valid_prediction,
                       test_prediction, loss, optimizer, accuracy, pre, lbl, beta_regu, prob=False):

        self.log.info("start transforming pairwise")

        train_data_left, train_data_right, train_labels_new = Utilities.transform_pairwise(train_dataset, train_labels, prob)
        valid_data_left, valid_data_right, valid_labels_new = Utilities.transform_pairwise(valid_dataset, valid_labels, prob)
        test_data_left, test_data_right, test_labels_new = Utilities.transform_pairwise(test_dataset, test_labels, prob)

        self.log.info("end transforming pairwise")

        with tf.Session(graph=graph, config=tf.ConfigProto(log_device_placement=True)) as session:
            session.run(tf.global_variables_initializer())
            logger.info('Initialized')
            for step in range(NNConfig.num_steps):
                offset = (step * NNConfig.batch_size) % (train_labels.shape[0] - NNConfig.batch_size)
                batch_data_left = train_data_left[offset:(offset + NNConfig.batch_size), :]
                batch_data_right = train_data_right[offset:(offset + NNConfig.batch_size), :]
                batch_labels = train_labels_new[offset:(offset + NNConfig.batch_size)]

                feed_dict = {tf_train_left: batch_data_left, tf_train_right: batch_data_right,
                             tf_train_labels: batch_labels}
                if NNConfig.regularization:
                    feed_dict[beta_regu] = NNConfig.beta_regu
                logger.debug('start optimizing')
                _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
                logger.debug('optimizing finished')
                if (step % NNConfig.summary_steps == 0):
                    logger.info("Minibatch loss at step %d: %f" % (step, l))
                    logger.info("Minibatch MSE: %.3f" % session.run(accuracy,
                                                                    feed_dict={pre: predictions, lbl: batch_labels}))
                    for i in range(0, 5):
                        print("label value:", batch_labels[i], \
                              "estimated value:", predictions[i])
                    # self.print_words(predictions, batch_labels)
                    steps, valid_left_batches, valid_right_batches, valid_label_batches = self.batchData_pairwise(data_left=valid_data_left,
                                                                                                                  data_right=valid_data_right,
                                                                                                                  labels=valid_labels_new,
                                                                                                                  batch_size=NNConfig.batch_size)
                    valid_mse = 0
                    for step in range(0, steps):
                        batch_prediction = session.run(valid_prediction,
                                                       feed_dict={tf_valid_left: valid_left_batches[step],
                                                                  tf_valid_right: valid_right_batches[step]})

                        batch_mse = session.run(accuracy, feed_dict={pre: batch_prediction,
                                                                     lbl: valid_label_batches[step]})
                        valid_mse += batch_mse
                    logger.info('Validation MSE: %.3f' % valid_mse)

                    steps, test_left_batches, test_right_batches, test_label_batches = self.batchData_pairwise(
                        data_left=test_data_left,
                        data_right=test_data_right,
                        labels=test_labels_new,
                        batch_size=NNConfig.batch_size)
                    test_mse = 0
                    for step in range(0, steps):
                        batch_prediction = session.run(test_prediction,
                                                       feed_dict={tf_test_left: test_left_batches[step],
                                                                  tf_test_right: test_right_batches[step],
                                                                  lbl: test_label_batches[step]})
                        batch_mse = session.run(accuracy, feed_dict={pre: batch_prediction,
                                                                     lbl: test_label_batches[step]})
                        test_mse += batch_mse
                    logger.info('Test MSE: %.3f' % test_mse)










