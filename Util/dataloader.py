import six.moves.cPickle as pickle
import sys
import os

sys.path.insert(0, os.path.abspath('..'))
import logging
from datautil import Retrieval_Data_Util, TextDataHandler, Utilities
from configs import DataConfig


class DataLoader(object):
    def __init__(self):
        logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                            datefmt='%d.%m.%Y %I:%M:%S %p', level=logging.INFO)
        self.log = logging.getLogger("Data Loader")
        self._d_handler = TextDataHandler(DataConfig.all_doc_path, DataConfig.save_dir_data)
        self._d_handler.truncate_vocab(DataConfig.vocab_size)
        self._r_datautil = Retrieval_Data_Util(DataConfig.run_path, DataConfig.qrel_path, DataConfig.qtitle_path)

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
        train_dataset, train_labels = dataset[start_tr:end_tr, :], labels[start_tr:end_tr, :]
        valid_dataset, valid_labels = dataset[start_v:end_v, ::], labels[start_v:end_v, :]
        test_dataset, test_labels = dataset[start_te:end_te, ::], labels[start_te:end_te, :]
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
        return self._d_handler.prepare_data(dts=dts, lbl=lbl,
                                            qid_title_dict=self._r_datautil.qid_title_dict)

    def get_ttv(self):
        pickle_file = os.path.join(DataConfig.save_dir_data, 'robust_binary_vec.pkl')
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
                self.create_datasets(dataset, labels, DataConfig.train_ratio, DataConfig.valid_ratio,
                                     DataConfig.test_ratio)
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
