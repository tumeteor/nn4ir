from abc import abstractmethod
import tensorflow as tf
class NN:
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
        data_batches = []
        labels_batches = []
        data_length = tf.shape(data)[0]
        steps = data_length % batch_size
        for step in steps:
            offset = (step * batch_size) % (data_length - batch_size)
            batch_data = data[offset:(offset + batch_size), :]
            batch_labels = labels[offset:(offset + batch_size)]
            batch_labels = batch_labels.reshape(data_length, 1)
            data_batches[step] = batch_data
            labels_batches[step] = batch_labels

        return steps, data_batches, labels_batches