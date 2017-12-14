import csv
import tensorflow as tf
class Evaluation:

    def loadGroundtruth(self, path):
        with open(path, 'rb') as f:
            csv_reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in csv_reader:
                label = row[0]
                url = row[1]



class Model:

    def __init__(self, model_path):
        self.session = tf.Session()

        tf.saved_model.loader.load(
            self.session,
            [tf.saved_model.tag_constants.SERVING],
            model_path)
        self.model= self.session.graph.get_tensor_by_name('final_ops/softmax:0')

    def predict(self, data):
        predictions = self.session.run(self.model, {'Placeholder:0': data})
        return predictions

