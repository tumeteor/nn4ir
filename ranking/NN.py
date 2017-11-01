from abc import abstractmethod
class NN:
    @abstractmethod
    def simple_NN(self,mode):
        pass

    def batchData(self, batch_size, data, labels):
        steps = len(data) % batch_size
        data_batches = []
        labels_batches = []
        for step in steps:
            offset = (step * batch_size) % (len(data) - batch_size)
            batch_data = data[offset:(offset + batch_size), :]
            batch_labels = labels[offset:(offset + batch_size)]
            batch_labels = batch_labels.reshape(len(batch_labels), 1)
            data_batches[step] = batch_data
            labels_batches[step] = batch_labels

        return steps, data_batches, labels_batches