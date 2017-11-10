import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath('..'))
import logging
from Util.datautil import Retrieval_Data_Util, TextDataHandler, Utilities
from Util.configs import NNConfig
import yaml
import os
with open(os.path.join(os.path.dirname(__file__),"configs.yaml")) as ymlfile:
    cfg = yaml.load(ymlfile)

class DataLoader(object):
    pretrained = False
    embedding = False
    twolabels = Utilities.twolabels
    def __init__(self, pretrained=False, embedding = False, qsize="1k"):
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
        else:
            self._data_config = cfg['DataConfig']
            self.log.info("using 1k queryset")
        self._d_handler = TextDataHandler(self._data_config['all_doc_path'], self._data_config['save_dir_data'])
        if not pretrained: self._d_handler.truncate_vocab(NNConfig.vocab_size)
        self._r_datautil = Retrieval_Data_Util(self._data_config['run_path'], self._data_config['qrel_path'], self._data_config['qtitle_path'])

        self.pretrained = pretrained
        self.embedding = embedding

        if pretrained:
            '''
            Load pretrained embeddings
            '''
            self.pretrain_vocab, self.embd = self.loadEmbedding(self._data_config['glove_file_path'])

    @property
    def d_handler(self):
        return self._d_handler

    @property
    def retrieval_data_uti(self):
        return self._rdu


    def create_datasets(self, dataset, labels, train_ratio, valid_ratio, test_ratio):
        data, label = Utilities.shufflize(dataset, labels)
        # split training, validation and testing sets
        train_size, valid_size, test_size = int(train_ratio * len(data)), int(valid_ratio * len(data)), int(
            test_ratio * len(data))
        start_tr, end_tr = 0, train_size - 1
        start_v, end_v = end_tr + 1, end_tr + valid_size
        start_te, end_te = end_v + 1, end_v + test_size
        train_dataset, train_labels = dataset[start_tr:end_tr, :], labels[start_tr:end_tr]
        valid_dataset, valid_labels = dataset[start_v:end_v, :], labels[start_v:end_v]
        test_dataset, test_labels = dataset[start_te:end_te, :], labels[start_te:end_te]
        print('Training:', train_dataset.shape, train_labels.shape)
        print('Validation:', valid_dataset.shape, valid_labels.shape)
        print('Testing:', test_dataset.shape, test_labels.shape)
        return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels

    def prepare_data(self):
        '''
        Prepare training data for NN
        :return:
        '''
        dts, lbl = self._r_datautil.get_pseudo_rel_qd_Bing(top_k=100)
        if self.pretrained:
            dataset, labels = self._d_handler.prepare_data_for_pretrained_embedding(dts, lbl, qid_title_dict=self._r_datautil.qid_title_dict)
            from tensorflow.contrib import learn
            vocab_processor = learn.preprocessing.VocabularyProcessor(NNConfig.max_doc_size)
            # fit the vocab from glove
            pretrain = vocab_processor.fit(self.pretrain_vocab)
            return np.array(list(vocab_processor.transform(dataset))), labels
        elif self.embedding:
            return self.d_handler.prepare_data_for_embedding_with_old_vocab(dts, lbl,
                                                                            qid_title_dict=self._r_datautil.qid_title_dict,
                                                                            length_max=NNConfig.max_doc_size)


        return self._d_handler.prepare_data(dts=dts, lbl=lbl,
                                            qid_title_dict=self._r_datautil.qid_title_dict)

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
                train_labels = save['train_labels']
                valid_dataset = save['valid_dataset']
                valid_labels = save['valid_labels']
                test_dataset = save['test_dataset']
                test_labels = save['test_labels']
                del save  # hint to help gc free up memory
                # assert train_dataset.shape()[1] == vocab_size, "vocabulary size in pickle files is not" + str(vocab_size)
                print('Training set', train_dataset.shape, train_labels.shape)
                print('Validation set', valid_dataset.shape, valid_labels.shape)
                print('Test set', test_dataset.shape, test_labels.shape)

        else:
            dataset, labels = self.prepare_data()
            train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = \
                self.create_datasets(dataset, labels, NNConfig.train_ratio, NNConfig.valid_ratio,
                                     NNConfig.test_ratio)
            try:
                f = open(pickle_file, 'wb')
                save = {
                    'train_dataset': train_dataset,
                    'train_labels': train_labels,
                    'valid_dataset': valid_dataset,
                    'valid_labels': valid_labels,
                    'test_dataset': test_dataset,
                    'test_labels': test_labels,
                }
                pickle.dump(save, f, protocol=4)
                f.close()
            except Exception as e:
                logging.error('Unable to save data to', pickle_file, ':', e)
                raise
            statinfo = os.stat(pickle_file)
            print('Compressed pickle size:', statinfo.st_size)
        return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels

    def loadEmbedding(self, embd_file_path):

        from gensim.models.keyedvectors import KeyedVectors
        model = KeyedVectors.load(embd_file_path)

        vocab = model.vocab
        embd = []
        for word in vocab:
            embd.append(model[word])

        return vocab, embd
