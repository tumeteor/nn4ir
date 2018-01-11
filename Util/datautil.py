# -*- coding: utf-8 -*-
import codecs
import fnmatch
import os
import random
import re
import sys
from collections import *
import nltk
import numpy as np
import tensorflow as tf
import six.moves.cPickle as pickle
import logging
from surt import surt
import csv
import itertools
import h5py
from math import ceil


# for long CSV
maxInt = sys.maxsize
decrement = True
while decrement:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.
    decrement = False
    try:
        csv.field_size_limit(maxInt)
    except OverflowError:
        maxInt = int(maxInt / 10)
        decrement = True


class MyError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class TextDataHandler:
    _force_read_input = False
    _word_to_id = {}
    _id_to_word = {}
    _words = []
    _unknown_terms = set()
    _unknown_terms_freq = 0

    _filenames = None

    def __init__(self, all_doc_path, save_dir, pretrained=False):
        logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                            datefmt='%d.%m.%Y %I:%M:%S %p', level=logging.INFO)
        self.log = logging.getLogger("Data Processing")
        self._filenames = Utilities.recursive_glob(all_doc_path, "*")
        Utilities.create_file_dir(save_dir)
        # base_name = os.path.basename(os.path.normpath(all_doc_path))
        # word_to_id
        if not pretrained:
            word_to_id_pickle = os.path.join(save_dir, 'word_to_id.pkl')
            self._word_to_id = self._maybe_pickling(word_to_id_pickle, self._build_dict_from_a_set_of_files, self._filenames)
            # id_to_word
            id_to_word_pickle = os.path.join(save_dir, 'id_to_word.pkl')
            self._id_to_word = self._maybe_pickling(id_to_word_pickle, self._build_reverse_dict, self._word_to_id)



    def _maybe_pickling(self, pickle_name, func, *func_param):
        if os.path.exists(pickle_name) and not self._force_read_input:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % pickle_name)
            try:
                with open(pickle_name, 'rb') as f:
                    data = pickle.load(f)
            except Exception as e:
                print('Unable to process data from', pickle_name, ':', e)
                raise
        else:
            data = func(*func_param)
            self.pickling(pickle_name, data)
        return data

    def pickling(self, pickle_file_name, data):
        print('Pickling %s' % pickle_file_name)
        try:
            with open(pickle_file_name, 'wb') as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            self.log.error('Unable to save data to', pickle_file_name, ':', e)

    @staticmethod
    def clean_str(string):
        """
        string cleaning
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        # string = re.sub(r"[^A-Za-z0-9]", " ", string)
        return string.strip().lower()

    def read_words(self, counter, filename, doc_count):
        """
        Tokenization using NLTK
        """
        with tf.gfile.GFile(filename, "r") as f:
            # refactor to read line by line
            for docline in f.readlines():
                # url \t doctext
                doc = docline.split("\t",1)[1]
                doc_tokens = nltk.word_tokenize(TextDataHandler.clean_str(doc),language='german')
                # print "number of tokens: {}".format(len(doc_tokens))
                counter.update(doc_tokens)
                doc_count += 1

                if doc_count % 500 == 0:
                    print(str(doc_count) + " docs has been processed")
            return counter, doc_count

    def make_arrays(self, nb_rows, vec_size):
        if nb_rows:
            dataset = np.ndarray((nb_rows, vec_size), dtype=np.float32)
            labels = np.ndarray((nb_rows, 2), dtype=np.float32)
        else:
            dataset, labels = None, None
        return dataset, labels


    def prepare_queries(self, queries, length_max, qid_title_dict):
        qv_list = np.zeros((len(queries),self.get_vocab_size()))
        i = 0
        for q in queries:
            # get title of q
            q = qid_title_dict[q]
            if i%10 == 0: print(q)
            q_tokens = nltk.word_tokenize(TextDataHandler.clean_str(q),language='german')
            q_wordIds_vec = self.word_list_to_id_list(q_tokens)
            qv_list[i] = self.get_binary_vector(q_wordIds_vec)
            i += 1
        return self.padding(qv_list, length_max)



    def prepare_data(self, dts, lbl, qid_title_dict):
        Bing_url_size = len(dts)
        if Bing_url_size != len(lbl):
            raise 'there is problem in the data...'
        docdict = {}

        '''
        Retrieve documents from files
        TODO: not so efficient
        '''
        for filename in self._filenames:
            with tf.gfile.GFile(filename, "r") as f:
                # refactor to read line by line
                for docline in f.readlines():
                    # url \t doctext
                    doc = docline.split("\t", 1)
                    docid = doc[0]
                    doc_tokens = nltk.word_tokenize(TextDataHandler.clean_str(doc[1]),language='german')

                    data_wordIds_vec = self.word_list_to_id_list(doc_tokens)
                    docdict[docid] = self.get_binary_vector(data_wordIds_vec)
        '''
        retrieve queries for documents (urls)
        '''
        nIns = 0
        for i in range(0, Bing_url_size - 1):
            # doc
            # check key both for docs and labels
            # Note: some times labels are missing :/
            if dts[i] in docdict.keys():
                # lbl[i]: queryid - rank pair
                if lbl[i][0] in qid_title_dict.keys():
                    nIns += 1


        dataset, labels = self.make_arrays(nIns, self.get_vocab_size())
        cnt = 0
        j = 0 # dataset idx
        for i in range (0, Bing_url_size -1):
            # doc
            # check key both for docs and labels
            # Note: some times labels are missing :/
            if dts[i] in docdict.keys():
                if lbl[i][0] in qid_title_dict.keys():
                    dataset[j] = docdict[dts[i]]
                else:
                    continue
            else:
                cnt += 1
                continue

            # query - label
            #label_tokens = nltk.word_tokenize(qid_title_dict[lbl[i]],language='german')
            #label_wordIds_vec = self.word_list_to_id_list(label_tokens)
            #labels[j] = self.get_binary_vector(label_wordIds_vec)
            labels[j] = [float(x) for x in lbl[i]]
            j += 1
        print("number of docs not in archive: {}".format(cnt))

        print('Full dataset tensor:', dataset.shape, labels.shape)
        # print('Mean:', np.mean(dataset))
        # print('Standard deviation:', np.std(dataset))
        return dataset, labels


    def prepare_data_for_pretrained_embedding(self, dts, lbl, qid_title_dict, testMode=False):

        Bing_url_size = len(dts)
        if Bing_url_size != len(lbl):
            raise 'there is problem in the data...'
        docdict = {}

        print("length of qid_title_dict: {}".format(len(qid_title_dict)))

        '''
        Retrieve documents from files
        TODO: not so efficient
        '''
        for filename in self._filenames:
            with tf.gfile.GFile(filename, "r") as f:
                # refactor to read line by line
                for docline in f.readlines():
                    # url \t doctext
                    doc = docline.split("\t", 1)
                    docid = doc[0]
                    docdict[docid] = TextDataHandler.clean_str(doc[1])
        '''
        retrieve queries for documents (urls)
        '''

        dataset = list()
        labels = list()
        # normalized Bing rank scores
        ranks = Utilities.ranknorm()
        for i in range(0, Bing_url_size - 1):
            # doc
            # check key both for docs and labels
            # Note: some times labels are missing :/
            # dts: urls
            if dts[i] in docdict.keys():
                if lbl[i][0] in qid_title_dict.keys():
                    dataset.append(docdict[dts[i]])
                else:
                    continue
            else:
                continue

            # query - label
            # labels.append(qid_title_dict[lbl[i]])
            rel_score = int(lbl[i][1]) if testMode else ranks[int(lbl[i][1])-1]
            if testMode:
                labels.append([lbl[i][0], rel_score, dts[i]])
            else:
                labels.append([lbl[i][0], rel_score])
        labels = np.array(labels)

        # print('Mean:', np.mean(dataset))
        # print('Standard deviation:', np.std(dataset))
        if testMode:
            return dataset, labels.reshape(len(labels), 3)
        else:
            return dataset, labels.reshape(len(labels), 2)

    def prepare_data_for_embedding_with_old_vocab(self, dts, lbl, qid_title_dict, length_max):
        Bing_url_size = len(dts)
        if Bing_url_size != len(lbl):
            raise 'there is problem in the data...'
        docdict = {}

        '''
        Retrieve documents from files
        TODO: not so efficient
        '''
        for filename in self._filenames:
            with tf.gfile.GFile(filename, "r") as f:
                # refactor to read line by line
                for docline in f.readlines():
                    # url \t doctext
                    doc = docline.split("\t", 1)
                    docid = doc[0]
                    doc_tokens = nltk.word_tokenize(TextDataHandler.clean_str(doc[1]), language='german')

                    data_wordIds_vec = self.word_list_to_id_list(doc_tokens)
                    docdict[docid] = data_wordIds_vec
        '''
        retrieve queries for documents (urls)
        '''
        dataset = list()
        labels = list()
        # normalized Bing rank scores
        ranks = Utilities.ranknorm()
        for i in range(0, Bing_url_size - 1):
            # doc
            # check key both for docs and labels
            # Note: some times labels are missing :/
            if dts[i] in docdict.keys():
                if lbl[i][0] in qid_title_dict.keys():
                    dataset.append(docdict[dts[i]])
                else:
                    continue
            else:
                continue

            # query - label
            #label_tokens = nltk.word_tokenize(qid_title_dict[lbl[i]], language='german')
            #label_wordIds_vec = self.word_list_to_id_list(label_tokens)
            #labels.append(label_wordIds_vec)
            rel_score = ranks[int(lbl[i][1]) - 1]
            labels.append([lbl[i][0], rel_score])

        # print('Full dataset tensor:', dataset.shape, labels.shape)
        # print('Mean:', np.mean(dataset))
        # print('Standard deviation:', np.std(dataset))
        padded_dataset = self.padding(dataset, length_max)
        labels = np.array(labels)

        return padded_dataset, labels.reshape(len(labels), 2)

    def padding(self, dataset, length_max):
        '''
        For tensors, pad with zeros
        :param dataset:
        :param length_max:
        :return:
        '''
        x_data = np.array(dataset)
        # truncate long docs
        x_data = [(x[0:length_max] if len(x) > length_max else x) for x in x_data]

        # Get lengths of each row of data
        lens = np.array([len(x_data[i]) for i in range(len(x_data))])

        # Mask of valid places in each row
        # mask = np.arange(lens.max()) < lens[:, None]
        mask = np.arange(length_max) < lens[:, None]

        # Setup output array and put elements from data into masked positions
        padded = np.zeros(mask.shape)
        padded[mask] = np.hstack((x_data[:]))
        return padded

    def _build_dict_from_a_set_of_files(self, filenames):
        '''
        Doc files
        :param filenames:
        :return:
        '''
        counter = Counter()
        cnt = 0
        for filename in filenames:
            # parse documents from file
            counter, cnt = self.read_words(counter,filename, cnt)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        # if vocab_size == -1:
        # 	vocab_size = len(count_pairs)
        # words, _ = list(zip(*count_pairs[:vocab_size]))
        words, _ = list(zip(*count_pairs))
        word_to_id = dict(zip(words, range(1, len(words) + 1)))
        return word_to_id

    def get_vocab_size(self):
        if not self._word_to_id.values:
            raise MyError('first build the dictionary...')
        return len(self._word_to_id) + 1  # unknown

    def truncate_vocab(self, new_vocab_size):
        if new_vocab_size < 0:
            print("New Vocab Size:", self.get_vocab_size())
            return
        self._word_to_id = OrderedDict(sorted(self._word_to_id.items(), key=lambda item: item[1])[:new_vocab_size - 1])
        self._id_to_word = self._build_reverse_dict(self._word_to_id)
        print("New Vocab Size:", self.get_vocab_size())

    def _build_reverse_dict(self, original_dict):
        if not original_dict:
            raise MyError('first build the dictionary...')
        reverse_dict = dict(zip(original_dict.values(), original_dict.keys()))
        return reverse_dict

    def get_id_of_word(self, word):
        #     return self._word_to_id.get(word, 0) # dictionary['UNK'] = 0
        if word in self._word_to_id:
            return self._word_to_id[word]
        else:  # dictionary['UNK'] = 0
            self._unknown_terms.add(word)
            self._unknown_terms_freq += 1
            return 0

    def get_word_of_id(self, i):
        if i == 0:
            word = 'UNK'
        elif i in self._id_to_word:
            word = self._id_to_word[i]
        else:
            print(i)
            raise MyError('unknown_id')
        return word

    def word_list_to_id_list(self, words):
        return [self.get_id_of_word(word) for word in words]

    def id_list_to_word_list(self, ids):
        return [self.get_word_of_id(i) for i in ids]

    def get_vocab_dict(self):
        return self._word_to_id.copy()

    def get_rev_vocab_dict(self):
        return self._id_to_word.copy()

    def get_vocab(self):
        if not self._words:
            self._words = list(self._word_to_id)
        return self._words

    def get_one_hot_vector(self, wordIds_vec):
        vocab_size = self.get_vocab_size()
        vec = (np.arange(vocab_size) == np.array(wordIds_vec)[:, None]).astype(np.float32)
        return np.array(vec)

    def get_ids_from_one_hot_vector(self, one_hot_Vec):
        # TODO: to be implimented
        return None

    def get_binary_vector(self, wordIds_vec):
        vec = np.zeros(self.get_vocab_size(), dtype=int)
        inices = list(set(wordIds_vec))
        vec[inices] = 1
        return np.array(vec)

    def get_ids_from_binary_vector(self, binary_vector):
        return np.nonzero(binary_vector)

    def get_freq_vector(self, wordIds_vec):
        vec = np.zeros(self.get_vocab_size(), dtype=int)
        counter = Counter(wordIds_vec)
        vec[list(counter.keys())] = list(counter.values())
        return np.array(vec)

    def get_ids_from_freq_vector(self, freq_vector):
        # return [[freq_vector[i]] * i for i in np.nonzero(freq_vector)]
        return None


class BatchCreator4RNN:
    def __init__(self, tensor, batch_size, seq_length):
        self.tensor = tensor
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.create_batches()
        self.pointer = 0

    # abcd -> x:abc y:bcd
    def create_batches(self):
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))
        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.tensor
        ydata = np.copy(self.tensor)

        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0


class Retrieval_Data_Util:
    # get relevance doc_query_pairs (from experts / crowdsourcing)
    # test: using Bing rank
    # TODO: for Bing rank: take top-50
    # query_id \t rank \t title \t description \t url \t query_id (tab delimiter)
    doc_query_pairs = []
    qid_title_dict = {}

    def __init__(self, runres, qrel, qtitles):
        with codecs.open(qtitles, "r", encoding='utf-8', errors='ignore') as f:
        #with open(qtitles, 'rb') as f:
            csv_reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in csv_reader:
            #for utf8_row in csv_reader:
                #row = [x.decode('utf8') for x in utf8_row]
                if len(row) < 2: continue
                self.qid_title_dict[row[0]] = row[1]
            f.close()
        with codecs.open(qrel, "r", encoding='utf-8', errors='ignore') as f:
        #with codecs.open(qrel, "rb") as f:
            csv_reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in csv_reader:
            #for utf8_row in csv_reader:
                # only select relevant document-query pairs
                #row = [x.decode('utf8') for x in utf8_row]
                # if row[3] == '1':
                # self.doc_query_pairs.append((row[2], row[0]))
                #if int(row[1]) <= 100:
                if len(row) < 5: continue
                self.doc_query_pairs.append((surt(row[4]), row[0]))

                # TODO: for now one-to-one query-url pair
                # self.doc_query_dict[surt(row[4])] = row[0]

            f.close()
        self._runRes = runres


    def get_rel_qd(self):
        d = [tup[0] for tup in self.doc_query_pairs]
        q = [tup[1] for tup in self.doc_query_pairs]
        return d, q

    def get_pseudo_rel_qd_Bing(self, top_k):
        '''
        query-doc relevance grade
        query_id \t rank \t title \t description \t url \t query_id (tab delimiter)
        :param top_k: top k documents
        :return:
        '''
        d = []
        q = []
        with codecs.open(self._runRes, "r", encoding='utf-8', errors='ignore') as f:
        #with open(self._runRes, "rb") as f:
            next(f) # skip the first line
            csv_reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in csv_reader:
            #for utf8_row in csv_reader:
                #row = [x.decode('utf8') for x in utf8_row]
                if len(row) <= 1:
                    print(row)
                    continue
                if int(row[1]) <= top_k:
                    # urls in Bing is not normalized yet
                    d.append(surt(row[4]))
                    # get Bing rank as label
                    # qid - rank
                    q.append([row[0],row[1]])
            f.close()
        return d, q

    def is_number(self,s):
        try:
            float(s)
            return True
        except ValueError:
            pass

        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass

        return False

    def loadGroundtruth(self):
        print(self._runRes)
        with open(self._runRes, 'r') as f:
            csv_reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            d = []
            q = []
            for row in csv_reader:
                if len(row) < 5: continue
                if not self.is_number(row[1]): continue
                q.append([row[0], row[1]]) # label
                d.append(surt(row[4])) # url
            f.close()
        return d,q

    def get_label(self, d, q):
        '''
        return binary relevance
        :param d:
        :param q:
        :return:
        '''
        return (d, q) in self.doc_query_pairs


class Utilities:


    twolabels = False

    @staticmethod
    def getIntancesFromQueries(queries, pairDict, qv, cache=False, h5py_path=None, type=None):
        newData = []
        newLabel = []
        newQueries = []
        i = 0
        for q in queries:
            docs = pairDict[q]
            for doc in docs:
                newData.append(doc[0])
                newLabel.append([(q).encode('utf8'), doc[1]])
                newQueries.append(qv[i])
            i += 1

        if not cache:
            return np.array(newData), np.array(newLabel), np.array(newQueries)

        else:
            #loop over h5py in batches
            batch_size = 10000
            batches_list = list(range(int(ceil(float(len(newData)) / batch_size))))
            with h5py.File(h5py_path, 'a') as hf:
                # loop over batches
                for n, i in enumerate(batches_list):
                    i_s = i * batch_size  # index of the first instance in the batch
                    i_e = min([(i + 1) * batch_size, len(newData)])  # index of the last instance in the batch

                    b_data = newData[i_s:i_e]
                    b_label = newLabel[i_s:i_e]
                    b_queries = newQueries[i_s:i_e]

                    if type + "_dataset" in hf.keys():
                        hf[type + "_dataset"].resize((hf[type + "_dataset"].shape[0] + len(b_data)), axis=0)
                        hf[type + "_dataset"][-len(b_data):] = b_data
                        hf[type + "_labels"].resize((hf[type + "_labels"].shape[0] + len(b_label)), axis=0)
                        hf[type + "_labels"][-len(b_label):] = b_label
                        hf[type + "_queryset"].resize((hf[type + "_queryset"].shape[0] + len(b_queries)), axis=0)
                        hf[type + "_queryset"][-len(b_queries):] = b_queries
                    else:
                        hf.create_dataset(type + "_dataset", data=b_data, maxshape=(None, len(b_data[0])),
                                          chunks=True,
                                          compression="gzip", compression_opts=6)
                        hf.create_dataset(type + "_labels", data=b_label, maxshape=(None, len(b_label[0])),
                                          chunks=True,
                                          compression="gzip", compression_opts=6)
                        hf.create_dataset(type + "_queryset", data=b_queries,
                                          maxshape=(None, len(b_queries[0])), chunks=True,
                                          compression="gzip", compression_opts=6)








    @staticmethod
    def shufflize(data, label, train_ratio, valid_ratio, test_ratio):
        assert len(data) == len(label)
        '''
        Shuffle per queries
        '''

        pairDict = Utilities.buildQueryDocDict(data, label, False)

        queries = np.array(list(pairDict.keys()))
        perm = np.random.permutation(len(queries))

        queries = queries[perm]


        train_size, valid_size, test_size = int(train_ratio * len(queries)), int(valid_ratio * len(queries)), int(
            test_ratio * len(queries))
        start_tr, end_tr = 0, train_size - 1
        start_v, end_v = end_tr + 1, end_tr + valid_size
        start_te, end_te = end_v + 1, end_v + test_size
        train_queries = queries[start_tr:end_tr]
        valid_queries = queries[start_v:end_v]
        test_queries = queries[start_te:end_te]
        return train_queries, valid_queries, test_queries, pairDict


    @staticmethod
    def recursive_glob(treeroot, pattern):
        results = []
        for base, dirs, files in os.walk(treeroot):
            goodfiles = fnmatch.filter(files, pattern)
            results.extend(os.path.join(base, f) for f in goodfiles)
        return results

    @staticmethod
    def create_file_dir(filename):
        if not os.path.exists(os.path.dirname(filename)):
            try:
                os.makedirs(os.path.dirname(filename))
            except OSError as exc:  # Guard against race condition
                if exc.errno != os.errno.EEXIST:
                    raise

    @staticmethod
    def sample_distribution(distribution):
        """Sample one element from a distribution assumed to be an array of normalized
    probabilities.
    """
        r = random.uniform(0, 1)
        s = 0
        for i in range(len(distribution)):
            s += distribution[i]
            if s >= r:
                return i
        return len(distribution) - 1

    @staticmethod
    def random_distribution(vector_size):
        """Generate a random column of probabilities."""
        b = np.random.uniform(0.0, 1.0, size=[1, vector_size])
        return b / np.sum(b, 1)[:, None]

    @staticmethod
    def accuracy(predictions, labels):
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
                / predictions.shape[0])

    @staticmethod
    def ranknorm():
        ranks = list(range(100,0,-1))
        r_norms = [Utilities.transform_rank(r) for r in ranks] if not Utilities.twolabels \
            else [Utilities.transform_bin_rank(r) for r in ranks]
            # else Utilities.softmax(ranks)

        return r_norms

    @staticmethod
    def transform_rank(r):
        if r < 10:
            r = 0
        elif 10 <= r <= 20:
            r = 0.1
        elif 20 <= r <= 30:
            r = 0.2
        elif 30 <= r <= 40:
            r = 0.3
        elif 40 <= r <= 50:
            r = 0.4
        elif 50 <= r <= 60:
            r = 0.5
        elif 60 <= r <= 70:
            r = 0.6
        elif 70 <= r <= 80:
            r = 0.7
        elif 80 <= r <= 90:
            r = 0.8
        elif 90 <= r <= 100:
            r = 0.9
        return r

    @staticmethod
    def tranform_binary_rank(r):
        if r < 50:
            r = 0
        else:
            r = 1
        return r

    @staticmethod
    def softmax(w, t=1.0):
        e = np.exp(np.array(w) / t)
        dist = e / np.sum(e)
        return dist

    @staticmethod
    def transform_ordinal(rank):
        rs = [0] * 10
        for i in range(0, len(rs)):
            if i <= rank * 10: rs[i] = 1
            else: rs[i] = 0
        return rs


    @staticmethod
    def transform_ordinal_rank(labels):
        ordinalLabels = np.empty((len(labels), 10))

        for i in range(0, len(labels)):
              ordinalLabels[i] = Utilities.transform_ordinal(float(labels[i][1]))

        return ordinalLabels


    @staticmethod
    def buildQueryDocDict2(data, labels, queries, eval):
        '''

        :param data: h5py file
        :param labels: h5py file
        :param queries: query vector, h5py file
        :param eval:
        :return:
        '''
        print('start')
        pair_dict = {}

        # loop over h5py in batches
        batch_size = 1000
        batches_list = list(range(int(ceil(float(len(data)) / batch_size))))
        # loop over batches
        for n, i in enumerate(batches_list):
            i_s = i * batch_size # index of the first instance in the batch
            i_e = min([(i + 1) * batch_size, len(data)]) # index of the last instance in the batch

            b_data = data[i_s:i_e, :]
            b_labels = labels[i_s:i_e, :]
            b_queries = queries[i_s:i_e, :]
            for j in range(0, len(b_data)):
                q = b_labels[j][0]
                url = None
                if eval:
                    url = b_labels[j][2]
                if q not in pair_dict:
                    pair_dict[q] = [[b_data[j], float(b_labels[j][1]), url, b_queries[j]]] if eval \
                        else [[b_data[j], float(b_labels[j][1]), b_queries[j]]]

                else:
                    if eval:
                        pair_dict[q].append([b_data[j], float(b_labels[j][1]), url, b_queries[j]])
                    else:
                        pair_dict[q].append([b_data[j], float(b_labels[j][1]), b_queries[j]])

        print('end')

        return pair_dict

    @staticmethod
    def buildQueryDocDict(data, labels, eval):
        '''

        :param data:
        :param labels:
        :param eval:
        :return:
        '''
        pair_dict = {}
        for i in range(0, len(data)):
            q = labels[i][0]
            url = None
            if eval:
                url = labels[i][2]
            if q not in pair_dict:
                pair_dict[q] = [[data[i], float(labels[i][1]), url]] if eval \
                    else [[data[i], float(labels[i][1])]]

            else:
                if eval:
                    pair_dict[q].append([data[i], float(labels[i][1]), url])
                else:
                    pair_dict[q].append([data[i], float(labels[i][1])])
        return pair_dict


    @staticmethod
    def transform_pairwise(data, labels, queries, h5py_path=None, prob=False, eval=False):
        """Transforms data into pairs with balanced labels for ranking"""

        assert len(data) == len(labels)
        assert len(data) == len(queries)
        print('start')
        pair_dict = Utilities.buildQueryDocDict2(data, labels, queries, eval)
        print('done')
        if not eval:

            print(data.shape)
            # data_left = np.empty((length, data.shape[1]))
            # data_right = np.empty((length, data.shape[1]))
            # label_new = np.empty((length, 1))

            # query - pair-wise combination scores
            qPmatDict = {}



            with h5py.File(h5py_path, 'a') as hf:
                data_left_batch = np.empty((10000,data.shape[1]))
                data_right_batch = np.empty((10000,data.shape[1]))
                data_queries_batch = np.empty((10000,queries.shape[1]))
                label_new_batch = np.empty((10000,1))
                cnt = 0
                idx = 0 #global buffer index

                for q, v in pair_dict.items():

                    # v: docs for q
                    if len(v) < 3: continue
                    # comb = itertools.combinations(range(len(v)), 2)
                    # npairs = 0
                    # for k, (i, j) in enumerate(comb):
                    #     # v: docid - label
                    #     if (v[i][0] == v[j][0]).all(): continue
                    #     npairs += 1

                    comb = itertools.permutations(range(len(v)), 2)
                    # data_left = np.empty((npairs, data.shape[1]))
                    # data_right = np.empty((npairs, data.shape[1]))
                    # label_new = np.empty((npairs, 1))
                    # data_queries = np.empty((npairs, queries.shape[1]))


                    for k, (i, j) in enumerate(comb):
                        # v: docid - label
                        if (v[i][0] == v[j][0]).all(): continue
                        if idx == 10000:
                            if "data_left" in hf.keys():
                                hf["data_left"].resize((hf["data_left"].shape[0] + data_left_batch.shape[0]), axis=0)
                                hf["data_left"][-data_left_batch.shape[0]:] = data_left_batch

                                hf["data_right"].resize((hf["data_right"].shape[0] + data_right_batch.shape[0]), axis=0)
                                hf["data_right"][-data_right_batch.shape[0]:] = data_right_batch

                                hf["label_new"].resize((hf["label_new"].shape[0] + label_new_batch.shape[0]), axis=0)
                                hf["label_new"][-label_new_batch.shape[0]:] = label_new_batch

                                hf["queries"].resize((hf["queries"].shape[0] + data_queries_batch.shape[0]), axis=0)
                                hf["queries"][-data_queries_batch.shape[0]:] = data_queries_batch

                                # reset batch arrays:
                                data_left_batch = np.empty((10000, data.shape[1]))
                                data_right_batch = np.empty((10000, data.shape[1]))
                                data_queries_batch = np.empty((10000, queries.shape[1]))
                                label_new_batch = np.empty((10000, 1))

                                print("query: {}".format(cnt))
                            else:
                                hf.create_dataset("data_left", data=data_left_batch, maxshape=(None, data.shape[1]),
                                                  chunks=True,
                                                  compression="gzip", compression_opts=6)
                                hf.create_dataset("data_right", data=data_right_batch, maxshape=(None, data.shape[1]),
                                                  chunks=True, compression="gzip", compression_opts=6)
                                hf.create_dataset("label_new", data=label_new_batch, maxshape=(None, 1), chunks=True,
                                                  compression="gzip", compression_opts=6)
                                hf.create_dataset("queries", data=data_queries_batch, maxshape=(None, queries.shape[1]),
                                                  chunks=True, compression="gzip", compression_opts=6)
                            idx = 0
                        data_left_batch[idx] = v[i][0]
                        data_right_batch[idx] = v[j][0]
                        if prob:
                            label_new_batch[idx] = v[i][1] / (v[i][1] + v[j][1]) if (v[i][1] + v[j][1]) != 0 else 0.5
                        else:
                            label_new_batch[idx] = v[i][1] - v[j][1]
                        data_queries_batch[idx] = v[i][2]
                        assert (v[i][2] == v[j][2]).all()
                        idx += 1

                    cnt += 1

        else:
            length = 0
            for q, v in pair_dict.items():
                if len(v[0]) < 3: continue
                comb = itertools.permutations(range(len(v)), 2)

                for k, (i, j) in enumerate(comb):
                    length += 1

            print(data.shape)
            data_left = np.empty((length, data.shape[1]))
            data_right = np.empty((length, data.shape[1]))
            label_new = np.empty((length, 1))
            data_queries = np.empty((length), queries.shape[1])

            idx = 0
            # query - pair-wise combination scores
            qPmatDict = {}
            for q, v in pair_dict.items():
                # v: docs for q
                if len(v[0]) < 3: continue
                comb = itertools.permutations(range(len(v)), 2)
                pmat = {}

                v = np.array(v)
                urls = v[:,2]
                for k, (i, j) in enumerate(comb):
                    data_left[idx] = v[i][0]
                    data_right[idx] = v[j][0]

                    label_new[idx] = 0 # meaningless for eval
                    data_queries[idx] = v[i][3]
                    pmat[idx] = (i, j)
                    idx += 1
                qPmatDict[q] = (pmat, urls)


            return data_left, data_right, data_queries, label_new, qPmatDict











