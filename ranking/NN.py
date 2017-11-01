from abc import abstractmethod
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