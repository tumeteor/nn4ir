from abc import abstractmethod
import numpy as np
import itertools


class NN:
    def __init__(self): pass

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
            batch_labels = batch_labels.reshape(batch_size, 1)
            data_batches.append(batch_data)
            labels_batches.append(batch_labels)

        return steps, data_batches, labels_batches



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

    @staticmethod
    def transform_pairwise(data, labels):
        """Transforms data into pairs with balanced labels for ranking"""

        assert len(data) == len(labels)
        data_left = []
        data_right = []
        label_new = []
        comb = itertools.combinations(len(data), 2)
        for k, (i, j) in enumerate(comb):
            if labels[i, 1] == labels[j, 1] or labels[i, 0] == labels[j, 0]:
                # skip same doc or different query
                continue

                data_left.append(data[i])
                data_right.append(data[j])
                labels.append(labels[i, 1] - labels[j, 1])

        return data_left, data_right, labels





