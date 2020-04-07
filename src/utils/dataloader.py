import sys
import six.moves.cPickle as pickle
import os
import numpy as np

sys.path.insert(0, os.path.abspath('..'))
import logging
from Util.datautil import Retrieval_Data_Util, TextDataHandler, Utilities
from Util.configs import NNConfig
import yaml
import h5py
import os

with open(os.path.join(os.path.dirname(__file__), "configs.yaml")) as ymlfile:
    cfg = yaml.load(ymlfile)


class DataLoader(object):
    pretrained = False
    embedding = False
    twolabels = Utilities.twolabels

    def __init__(self, pretrained=False, embedding=False, qsize="1k"):
        logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                            datefmt='%d.%m.%Y %I:%M:%S %p', level=logging.INFO)
        self.log = logging.getLogger("Data Loader")
        if qsize == "1k":
            self._data_config = cfg['DataConfig']
            self.log.info("using 1k queryset")
        elif qsize == "10k":
            self._data_config = cfg['DataConfig10k']
            self.log.info("using 10k queryset")
        elif qsize == "100k":
            self._data_config = cfg['DataConfig100k']
            self.log.info("using 100k queryset")
        elif qsize == "15":
            self._data_config = cfg['TestConfig']
            self.log.info("using test set")
        else:
            self._data_config = cfg['DataConfig']
            self.log.info("using 1k queryset")
        self._d_handler = TextDataHandler(self._data_config['all_doc_path'], self._data_config['save_dir_data'],
                                          pretrained)
        if not pretrained: self._d_handler.truncate_vocab(NNConfig.vocab_size)
        self._r_datautil = Retrieval_Data_Util(self._data_config['run_path'], self._data_config['qrel_path'],
                                               self._data_config['qtitle_path'])

        self.pretrained = pretrained
        self.embedding = embedding

        if pretrained:
            '''
            Load pretrained embeddings
            '''
            self.pretrain_vocab, self.embd = self.load_embedding(self._data_config['glove_file_path'])

            from tensorflow.contrib import learn
            self.vocab_processor = learn.preprocessing.VocabularyProcessor(NNConfig.max_doc_size)
            # fit the vocab from glove
            pretrain = self.vocab_processor.fit(self.pretrain_vocab)

    @property
    def d_handler(self):
        return self._d_handler

    @property
    def retrieval_data_uti(self):
        return self._rdu

    def create_datasets2(self, dataset, labels, train_ratio, valid_ratio, test_ratio,
                         h5py_path):
        # shuffle by queries
        train_queries, valid_queries, test_queries, pairDict = Utilities.shufflize(dataset, labels, train_ratio,
                                                                                   valid_ratio, test_ratio)
        # split training, validation and testing sets

        if self.pretrained:
            train_qv = np.array(list(self.vocab_processor.transform(train_queries)))
            valid_qv = np.array(list(self.vocab_processor.transform(valid_queries)))
            test_qv = np.array(list(self.vocab_processor.transform(test_queries)))
        else:
            train_qv = self._d_handler.prepare_queries(train_queries, length_max=NNConfig.max_doc_size,
                                                       qid_title_dict=self._r_datautil.qid_title_dict)
            valid_qv = self._d_handler.prepare_queries(valid_queries, length_max=NNConfig.max_doc_size,
                                                       qid_title_dict=self._r_datautil.qid_title_dict)
            test_qv = self._d_handler.prepare_queries(test_queries, length_max=NNConfig.max_doc_size,
                                                      qid_title_dict=self._r_datautil.qid_title_dict)

        Utilities.get_intances_from_queries(train_queries, pairDict, train_qv, cache=True, h5py_path=h5py_path,
                                            type="train")
        Utilities.get_intances_from_queries(valid_queries, pairDict, valid_qv, cache=True, h5py_path=h5py_path,
                                            type="valid")
        Utilities.get_intances_from_queries(test_queries, pairDict, test_qv, cache=True, h5py_path=h5py_path, type="test")

    def create_datasets(self, dataset, labels, train_ratio, valid_ratio, test_ratio):
        # shuffle by queries
        train_queries, valid_queries, test_queries, pairDict = Utilities.shufflize(dataset, labels, train_ratio,
                                                                                   valid_ratio, test_ratio)
        # split training, validation and testing sets

        if self.pretrained:
            train_qv = np.array(list(self.vocab_processor.transform(train_queries)))
            valid_qv = np.array(list(self.vocab_processor.transform(valid_queries)))
            test_qv = np.array(list(self.vocab_processor.transform(test_queries)))
        else:
            train_qv = self._d_handler.prepare_queries(train_queries, length_max=NNConfig.max_doc_size,
                                                       qid_title_dict=self._r_datautil.qid_title_dict)
            valid_qv = self._d_handler.prepare_queries(valid_queries, length_max=NNConfig.max_doc_size,
                                                       qid_title_dict=self._r_datautil.qid_title_dict)
            test_qv = self._d_handler.prepare_queries(test_queries, length_max=NNConfig.max_doc_size,
                                                      qid_title_dict=self._r_datautil.qid_title_dict)

        train_dataset, train_labels, train_queries = Utilities.get_intances_from_queries(train_queries, pairDict, train_qv)
        valid_dataset, valid_labels, valid_queries = Utilities.get_intances_from_queries(valid_queries, pairDict, valid_qv)
        test_dataset, test_labels, test_queries = Utilities.get_intances_from_queries(test_queries, pairDict, test_qv)

        return train_dataset, train_labels, train_queries, valid_dataset, valid_labels, valid_queries, \
               test_dataset, test_labels, test_queries

    def prepare_data(self):
        '''
        Prepare training data for NN
        :return:
        '''
        dts, lbl = self._r_datautil.get_pseudo_rel_qd_bing(top_k=100)
        if self.pretrained:
            print("load vocab from pretrained.")
            dataset, labels = self._d_handler.prepare_data_for_pretrained_embedding(dts, lbl,
                                                                                    qid_title_dict=self._r_datautil.qid_title_dict)

            return np.array(list(self.vocab_processor.transform(dataset))), labels
        elif self.embedding:
            print("prepare data for self embedding.")
            return self.d_handler.prepare_data_for_embedding_with_old_vocab(dts, lbl,
                                                                            qid_title_dict=self._r_datautil.qid_title_dict,
                                                                            length_max=NNConfig.max_doc_size)

        return self._d_handler.prepare_data(dts=dts, lbl=lbl,
                                            qid_title_dict=self._r_datautil.qid_title_dict)

    def prepare_test_data(self):
        dts, lbl = self._r_datautil.load_groundtruth()
        print("test size: {}".format(len(dts)))
        if self.pretrained:
            dataset, labels = self._d_handler.prepare_data_for_pretrained_embedding(dts, lbl,
                                                                                    qid_title_dict=self._r_datautil.qid_title_dict,
                                                                                    testMode=True)
            print("test size2: {}".format(len(dataset)))
            from tensorflow.contrib import learn
            vocab_processor = learn.preprocessing.VocabularyProcessor(NNConfig.max_doc_size)
            # fit the vocab from glove
            pretrain = vocab_processor.fit(self.pretrain_vocab)
            print(labels[:, 0])
            return np.array(list(vocab_processor.transform(dataset))), labels, np.array(
                list(vocab_processor.transform(labels[:, 0])))
        elif self.embedding:
            '''
            TODO
            '''
            pass

    def get_ttv(self):
        if self.pretrained:
            pickle_file = os.path.join(self._data_config['save_dir_data'], 'robust_pre_vec.pkl')
        elif self.embedding:
            if self.twolabels:
                pickle_file = os.path.join(self._data_config['save_dir_data'], 'robust_emb_tmp.pkl')
            else:
                pickle_file = os.path.join(self._data_config['save_dir_data'], 'robust_emb_vec.pkl')
        else:
            pickle_file = os.path.join(self._data_config['save_dir_data'], 'robust_binary_vec.pkl')

        if os.path.exists(pickle_file):
            print('%s already present - Skipping pickling.' % pickle_file)
            with open(pickle_file, 'rb') as f:
                save = pickle.load(f, encoding="bytes")
                train_dataset = save['train_dataset']
                print(train_dataset[1:10, :])
                train_labels = save['train_labels']
                train_queries = save['train_queries']
                print(train_queries[1:10, :])
                valid_dataset = save['valid_dataset']
                valid_labels = save['valid_labels']
                valid_queries = save['valid_queries']
                test_dataset = save['test_dataset']
                test_labels = save['test_labels']
                test_queries = save['test_queries']
                del save  # hint to help gc free up memory
                assert train_dataset.shape()[1] == vocab_size, "vocabulary size in pickle files is not" + str(vocab_size)
                print('Training set', train_dataset.shape, train_labels.shape, train_queries.shape)
                print('Validation set', valid_dataset.shape, valid_labels.shape, valid_queries.shape)
                print('Test set', test_dataset.shape, test_labels.shape, test_labels.shape)

        else:
            dataset, labels = self.prepare_data()
            train_dataset, train_labels, train_queries, valid_dataset, valid_labels, valid_queries, \
            test_dataset, test_labels, test_queries = \
                self.create_datasets(dataset, labels, NNConfig.train_ratio, NNConfig.valid_ratio, NNConfig.test_ratio)
            try:
                f = open(pickle_file, 'wb')
                save = {
                    'train_dataset': train_dataset,
                    'train_labels': train_labels,
                    'train_queries': train_queries,
                    'valid_dataset': valid_dataset,
                    'valid_labels': valid_labels,
                    'valid_queries': valid_queries,
                    'test_dataset': test_dataset,
                    'test_labels': test_labels,
                    'test_queries': test_queries
                }
                pickle.dump(save, f, protocol=4)
                f.close()
            except Exception as e:
                logging.error('Unable to save data to', pickle_file, ':', e)
                raise
            statinfo = os.stat(pickle_file)
            print('Compressed pickle size:', statinfo.st_size)
        return train_dataset, train_labels, train_queries, valid_dataset, valid_labels, valid_queries, test_dataset, test_labels, test_queries

    def get_ttv2(self):
        print("use cache mode for preprocessing/loading large data")
        if self.pretrained:
            pickle_file = os.path.join(self._data_config['save_dir_data'], 'robust_pre_vec.h5')
        elif self.embedding:
            if self.twolabels:
                pickle_file = os.path.join(self._data_config['save_dir_data'], 'robust_emb_tmp.h5')
            else:
                pickle_file = os.path.join(self._data_config['save_dir_data'], 'robust_emb_vec.h5')
        else:
            pickle_file = os.path.join(self._data_config['save_dir_data'], 'robust_binary_vec.h5')

        if not os.path.exists(pickle_file):

            dataset, labels = self.prepare_data()
            self.create_datasets2(dataset, labels, NNConfig.train_ratio, NNConfig.valid_ratio,
                                  NNConfig.test_ratio, pickle_file)
        else:
            print('%s already present - Skipping pickling.' % pickle_file)
        f = h5py.File(pickle_file, 'r')
        for key in f.keys():
            print(key)
        train_dataset = f.get('train_dataset')
        train_labels = f.get('train_labels')
        train_queryset = f.get('train_queryset')

        valid_dataset = f.get('valid_dataset')
        valid_labels = f.get('valid_labels')
        valid_queryset = f.get('valid_queryset')

        test_dataset = f.get('test_dataset')
        test_labels = f.get('test_labels')
        test_queryset = f.get('test_queryset')
        assert train_dataset.shape()[1] == vocab_size, "vocabulary size in pickle files is not" + str(vocab_size)
        return train_dataset, train_labels, train_queryset, valid_dataset, valid_labels, valid_queryset, \
               test_dataset, test_labels, test_queryset

    @staticmethod
    def load_embedding(embd_file_path):

        from gensim.models.keyedvectors import KeyedVectors
        model = KeyedVectors.load(embd_file_path)

        vocab = model.vocab
        embd = []
        for word in vocab:
            embd.append(model[word])

        return vocab, embd

    def load_test_data(self):
        """
        For evaluation purpose
        :return:
        """
        if self.pretrained:
            pickle_file = os.path.join(self._data_config['save_dir_data'], 'test_pre_vec.pkl')
        elif self.embedding:
            if self.twolabels:
                pickle_file = os.path.join(self._data_config['save_dir_data'], 'test_emb_tmp.pkl')
            else:
                pickle_file = os.path.join(self._data_config['save_dir_data'], 'test_emb_vec.pkl')
        else:
            pickle_file = os.path.join(self._data_config['save_dir_data'], 'test_binary_vec.pkl')

        if os.path.exists(pickle_file):
            print('%s already present - Skipping pickling.' % pickle_file)
            with open(pickle_file, 'rb') as f:
                save = pickle.load(f, encoding="bytes")
                dataset = save['test_dataset']
                labels = save['test_label']
                queries = save['test_queries']
        else:
            dataset, labels, queries = self.prepare_test_data()
            try:
                f = open(pickle_file, 'wb')
                save = {
                    'test_dataset': dataset,
                    'test_label': labels,
                    'test_queries': queries
                }
                pickle.dump(save, f, protocol=4)
                f.close()
            except Exception as e:
                logging.error('Unable to save data to', pickle_file, ':', e)
                raise

            statinfo = os.stat(pickle_file)
            print('Compressed pickle size:', statinfo.st_size)
        return dataset, labels, queries
