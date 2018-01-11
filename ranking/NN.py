from abc import abstractmethod
import numpy as np
import itertools
import logging
import tensorflow as tf
from Util.configs import NNConfig
from Util.datautil import Utilities
import os
import yaml
import h5py
from pathlib import Path
logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
fileHandler = logging.FileHandler("{0}/{1}.log".format("./", "Fullyconnected_NN"))
fileHandler.setFormatter(logFormatter)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(fileHandler)
logger.addHandler(consoleHandler)
with open(os.path.abspath(os.path.join(os.path.dirname(__file__),"../Util","configs.yaml"))) as ymlfile:
    cfg = yaml.load(ymlfile)
class NN:

    def __init__(self):

        self.train_dataset = None
        self.train_labels = None
        self.train_queries = None
        self.valid_dataset = None
        self.valid_labels = None
        self.valid_queries = None
        self.test_dataset = None
        self.test_labels = None
        self.test_queries = None
        self.log = logger
        self.pretrained = False
        self.hybrid = False
        self.d_loader = None
        self.trainwriter = open('tmp/trainloss-10kordinal-pointwise.tsv', 'w', buffering=1)
        self.validwriter = open('tmp/validloss-10kordinal-pointwise.tsv','w', buffering=1)
        self.testwriter = open('tmp/testloss-10kordinal-pointwise.tsv','w', buffering=1)

        self.ordinal = False

    @abstractmethod
    def simple_NN(self):
        pass

    @abstractmethod
    def simple_NN_pairwise(self):
        pass

    @abstractmethod
    def simple_NN_prob(self):
        pass

    def batchData(self, batch_size, data, labels, queries):
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
        queries_batches = list()
        for step in range(0, steps):
            offset = (step * batch_size) % (data_length - batch_size)
            batch_data = data[offset:(offset + batch_size), :]
            batch_labels = labels[offset:(offset + batch_size)]
            batch_queries = queries[offset:(offset + batch_size)]
            if not self.ordinal:
                # get rank only from label
                batch_labels = batch_labels[:, 1]
                batch_labels = batch_labels.reshape(batch_size, 1)

            data_batches.append(batch_data)
            labels_batches.append(batch_labels)
            queries_batches.append(batch_queries)

        return steps, data_batches, labels_batches, queries_batches

    @staticmethod
    def batchData_pairwise(batch_size, data_left, data_right, labels, queries):
        '''
        split data into smaller batch by the first tensor dimension
        :param batch_size:
        :param data: Tensor
        :param labels:
        :return:
        '''
        assert (len(data_left) == len(data_right))
        assert(len(data_left) == len(queries))
        data_length = len(data_left)
        steps = int(data_length / batch_size)
        data_left_batches = list()
        data_right_batches = list()
        labels_batches = list()
        queries_batches = list()
        for step in range(0, steps):
            offset = (step * batch_size) % (data_length - batch_size)
            batch_data_left = data_left[offset:(offset + batch_size), :]
            batch_data_right = data_left[offset:(offset + batch_size), :]
            batch_labels = labels[offset:(offset + batch_size)]
            batch_queries= queries[offset:(offset + batch_size)]
            # get rank only from label
            batch_labels = batch_labels.reshape(batch_size, 1)
            data_left_batches.append(batch_data_left)
            data_right_batches.append(batch_data_right)
            labels_batches.append(batch_labels)
            queries_batches.append(batch_queries)

        return steps, data_left_batches, data_right_batches, labels_batches, queries_batches

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



    def train(self, graph, tf_train_dataset, tf_train_labels, tf_train_queries,
              tf_valid_dataset, tf_valid_queries, tf_test_dataset, tf_test_queries, train_prediction, valid_prediction,
              test_prediction, loss, optimizer, accuracy, pre, lbl, beta_regu):
        with tf.Session(graph=graph, config=tf.ConfigProto(log_device_placement=True)) as session:
            session.run(tf.global_variables_initializer())
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            if self.pretrained:
                session.run(self.embedding_init, feed_dict={self.embedding_placeholder: self.embedding})
            logger.info('Initialized')

            if self.ordinal:
                self.train_labels = Utilities.transform_ordinal_rank(self.train_labels)
                self.valid_labels = Utilities.transform_ordinal_rank(self.valid_labels)
                self.test_labels = Utilities.transform_ordinal_rank(self.test_labels)

            vsteps, valid_data_batches, valid_label_batches, valid_query_batches = self.batchData(
                data=self.valid_dataset,
                labels=self.valid_labels,
                queries=self.valid_queries,
                batch_size=NNConfig.batch_size)

            tsteps, test_data_batches, test_label_batches, test_query_batches = self.batchData(data=self.test_dataset,
                                                                                               labels=self.test_labels,
                                                                                               queries=self.test_queries,
                                                                                               batch_size=NNConfig.batch_size)

            for step in range(NNConfig.num_steps):
                offset = (step * NNConfig.batch_size) % (self.train_labels.shape[0] - NNConfig.batch_size)
                batch_data = self.train_dataset[offset:(offset + NNConfig.batch_size), :]
                batch_labels = self.train_labels[offset:(offset + NNConfig.batch_size)]
                batch_queries = self.train_queries[offset:(offset + NNConfig.batch_size), :]

                if self.ordinal: pass
                else:
                    # get rank only from label
                    batch_labels = batch_labels[:, 1]
                    batch_labels = batch_labels.reshape(len(batch_labels), 1)
                    batch_labels = batch_labels.astype(float)

                # print('-' * 80)
                # for vec in batch_labels:
                #   print('.' * 200)
                #   print(self.get_words(vec))

                feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, tf_train_queries: batch_queries}
                if NNConfig.regularization:
                    feed_dict[beta_regu] = NNConfig.beta_regu
                logger.debug('start optimizing')
                _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
                logger.debug('optimizing finished')
                if (step % NNConfig.summary_steps == 0):
                    logger.info("Minibatch loss at step %d: %f" % (step, l))
                    self.trainwriter.write(str(step) + "\t" + '%.3f' % (l))
                    self.trainwriter.write("\n")
                    # logger.info("Minibatch MSE: %.3f" % session.run(accuracy,
                    #                                                 feed_dict={pre: predictions, lbl: batch_labels}))
                    for i in range(0, 10):
                        print("label value:", batch_labels[i], \
                              "estimated value:", predictions[i])
                    # self.print_words(predictions, batch_labels)
                    valid_mse = 0
                    for step in range(0, vsteps):
                        batch_prediction = session.run(valid_prediction,
                                                       feed_dict={tf_valid_dataset: valid_data_batches[step],
                                                                  tf_valid_queries:valid_query_batches[step]})

                        batch_mse = session.run(accuracy, feed_dict={pre: batch_prediction,
                                                                     lbl: valid_label_batches[step]})
                        valid_mse += batch_mse
                    logger.info('Validation loss: %.3f' % (valid_mse))
                    self.validwriter.write(str(step) + "\t" + '%.3f' % (valid_mse / float(vsteps)))
                    self.validwriter.write("\n")

                    test_mse = 0
                    for step in range(0, tsteps):
                        batch_prediction = session.run(test_prediction,
                                                       feed_dict={tf_test_dataset: test_data_batches[step],
                                                                  tf_test_queries: test_query_batches[step]})
                        batch_mse = session.run(accuracy, feed_dict={pre: batch_prediction,
                                                                     lbl: test_label_batches[step]})
                        test_mse += batch_mse
                    logger.info('Test loss: %.3f' % (test_mse))
                    self.testwriter.write(str(step) + "\t" + '%.3f' % (test_mse))
                    self.testwriter.write("\n")

            self.trainwriter.close()
            self.validwriter.close()
            self.testwriter.close()

            save_path = saver.save(session, "models/model-pointwise.ckpt")
            print("Model saved in file: %s" % save_path)


    def train_pairwise(self, graph, train_dataset, train_labels, train_queries, valid_dataset, valid_labels, valid_queries,
                       test_dataset, test_labels, test_queries,
                       tf_train_left, tf_train_right,
                       tf_train_labels, tf_train_queries,
                       tf_valid_left, tf_valid_right, tf_valid_queries,
                       tf_test_left, tf_test_right, tf_test_queries,
                       train_prediction, valid_prediction,
                       test_prediction, loss, optimizer, accuracy, pre, lbl, beta_regu, prob=False):

        self.log.info("start transforming pairwise")
        train_h5_path = self.d_loader._data_config['h5py_file_path']+"/train_pw_tmp.h5.lzf"
        if not Path(train_h5_path).exists():
            Utilities.transform_pairwise(train_dataset, train_labels, train_queries, train_h5_path, prob)
        train_hf = h5py.File(train_h5_path,'r')
        train_data_left = train_hf.get('data_left')
        train_data_right = train_hf.get('data_right')
        train_labels_new = train_hf.get('label_new')
        train_data_queries = train_hf.get('queries')
        self.log.info("end train")
        valid_h5_path = self.d_loader._data_config['h5py_file_path']+"/valid_pw_tmp.h5.lzf"
        if not Path(valid_h5_path).exists():
            Utilities.transform_pairwise(valid_dataset, valid_labels, valid_queries, valid_h5_path, prob)
        valid_hf = h5py.File(valid_h5_path, 'r')
        valid_data_left = valid_hf.get('data_left')
        valid_data_right = valid_hf.get('data_right')
        valid_labels_new = valid_hf.get('label_new')
        valid_data_queries = valid_hf.get('queries')
        self.log.info("end valid")
        test_h5_path = self.d_loader._data_config['h5py_file_path']+"/test_pw_tmp.h5.lzf"
        if not Path(test_h5_path).exists():
            Utilities.transform_pairwise(test_dataset, test_labels, test_queries, test_h5_path, prob)
        test_hf = h5py.File(test_h5_path,'r')
        test_data_left = test_hf.get('data_left')
        test_data_right = test_hf.get('data_right')
        test_labels_new = test_hf.get('label_new')
        test_data_queries = test_hf.get('queries')


        self.log.info("end transforming pairwise")

        with tf.Session(graph=graph, config=tf.ConfigProto(log_device_placement=True)) as session:
            session.run(tf.global_variables_initializer())
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            logger.info('Initialized')
            # vsteps, valid_left_batches, valid_right_batches, valid_label_batches, valid_query_batches = NN.batchData_pairwise(
            #     data_left=valid_data_left,
            #     data_right=valid_data_right,
            #     labels=valid_labels_new,
            #     queries=valid_data_queries,
            #     batch_size=NNConfig.batch_size)
            # tsteps, test_left_batches, test_right_batches, test_label_batches, test_query_batches = NN.batchData_pairwise(
            #     data_left=test_data_left,
            #     data_right=test_data_right,
            #     labels=test_labels_new,
            #     queries=test_data_queries,
            #     batch_size=NNConfig.batch_size)
            for step in range(NNConfig.num_steps):
                offset = (step * NNConfig.batch_size) % (train_labels_new.shape[0] - NNConfig.batch_size)
                batch_data_left = train_data_left[offset:(offset + NNConfig.batch_size), :]
                batch_data_right = train_data_right[offset:(offset + NNConfig.batch_size), :]
                batch_labels = train_labels_new[offset:(offset + NNConfig.batch_size)]
                batch_queries = train_data_queries[offset:(offset + NNConfig.batch_size), :]
                # get rank only from label
                #batch_labels = batch_labels[:, 1]
                batch_labels = batch_labels.reshape(len(batch_labels), 1)

                feed_dict = {tf_train_left: batch_data_left, tf_train_right: batch_data_right,
                             tf_train_labels: batch_labels, tf_train_queries: batch_queries}
                if NNConfig.regularization:
                    feed_dict[beta_regu] = NNConfig.beta_regu
                logger.debug('start optimizing')
                _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
                logger.debug('optimizing finished')
                if (step % 100 == 0): print(step)
                if (step % NNConfig.summary_steps == 0):
                    logger.info("Minibatch loss at step %d: %f" % (step, l))
                    self.trainwriter.write(str(step) + "\t" + '%.3f' % (l))
                    self.trainwriter.write("\n")

                    # logger.info("Minibatch MSE: %.3f" % session.run(accuracy,
                    #                                                 feed_dict={pre: predictions, lbl: batch_labels}))
                    for i in range(0, 5):
                        print("label value:", batch_labels[i], \
                              "estimated value:", predictions[i])
                    # self.print_words(predictions, batch_labels)

                    valid_mse = 0
                    # for vstep in range(0, vsteps):
                    #     batch_prediction = session.run(valid_prediction,
                    #                                    feed_dict={tf_valid_left: valid_left_batches[vstep],
                    #                                               tf_valid_right: valid_right_batches[vstep],
                    #                                               tf_valid_queries: valid_query_batches[vstep]})
                    #
                    #     if (vstep % NNConfig.summary_steps == 0):
                    #         for i in range(0, 5):
                    #             print("vlabel value:", valid_label_batches[vstep][i], \
                    #                   "vestimated value:", batch_prediction[i])
                    #
                    #
                    #     batch_mse = session.run(accuracy, feed_dict={pre: batch_prediction,
                    #                                                  lbl: valid_label_batches[vstep]})
                    #     valid_mse += batch_mse
                    # logger.info('Validation loss: %.3f' % (valid_mse))
                    # self.validwriter.write(str(step) + "\t" + '%.3f' % (valid_mse))
                    # self.validwriter.write("\n")
                    #
                    #
                    # test_mse = 0
                    # for step in range(0, tsteps):
                    #     batch_prediction = session.run(test_prediction,
                    #                                    feed_dict={tf_test_left: test_left_batches[step],
                    #                                               tf_test_right: test_right_batches[step],
                    #                                               tf_test_queries: test_query_batches[step]})
                    #     batch_mse = session.run(accuracy, feed_dict={pre: batch_prediction,
                    #                                                  lbl: test_label_batches[step]})
                    #     test_mse += batch_mse
                    # logger.info('Test loss: %.3f' % (test_mse))
                    # self.testwriter.write(str(step) + "\t" + '%.3f' % (test_mse))
                    # self.testwriter.write("\n")


            self.trainwriter.close()
            self.validwriter.close()
            self.testwriter.close()

            save_path = saver.save(session, "models/model-pairwise-tmp.ckpt")
            print("Model saved in file: %s" % save_path)








