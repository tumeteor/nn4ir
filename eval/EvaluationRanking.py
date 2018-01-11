from Util.dataloader import DataLoader
from ranking.NN import NNConfig, NN
import tensorflow as tf
from metrics.rank_metrics import ndcg_at_k
from argparse import ArgumentParser
import numpy as np
from Util.datautil import Utilities
class Evaluation:
    def __init__(self, embedding=False, pretrained=False):
        print("Eval init.")
        self.d_loader = DataLoader(embedding=embedding, pretrained=pretrained, qsize="15")
        self.test_dataset, self.test_labels, self.test_queries = self.d_loader.loadTestData()


    def getLblDict(self):
        lblDict = {}
        i = 0
        for row in self.test_labels:
            # row[0] = q
            if row[0] in lblDict:
                # row[2] = url
                # row[1] = label
                lblDict[row[0]][row[2]] = row[1]
            else:
                urlLabels = {}
                urlLabels[row[2]] = row[1]
                lblDict[row[0]] = urlLabels

        return lblDict


    def predict_pointwise(self, model_path):
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        with tf.Session() as session:
            # Restore variables from disk.
            saver.restore(session, model_path)
            print("Model restored.")
            graph = tf.get_default_graph()
            test_prediction = graph.get_tensor_by_name('test_prediction:0')
            tf_test_dataset = graph.get_tensor_by_name('test:0')
            tf_test_queries = graph.get_tensor_by_name('test_queries:0')

            vsteps, test_data_batches, test_label_batches, test_query_batches = NN.batchData(data=self.test_dataset,
                                                                         labels=self.test_labels,
                                                                         queries=self.test_queries,
                                                                         batch_size=NNConfig.batch_size)

            predictions = []
            for step in range(0, vsteps):
                batch_predictions = session.run(test_prediction, feed_dict={tf_test_dataset: test_data_batches[step],
                                                                            tf_test_queries: test_query_batches[step]})
                predictions.append(batch_predictions)

            assert len(predictions) == len(self.test_labels)
            return predictions

    def mle(self, pmat, max_iter=100):
        n = pmat.shape[0] #number of items
        wins = np.sum(pmat, axis=0)
        params = np.ones(n, dtype=float)
        for _ in range(max_iter):
            tiled = np.tile(params, (n, 1))
            combined = 1.0 / (tiled + tiled.T)
            np.fill_diagonal(combined, 0)
            nxt = wins / np.sum(combined, axis=0)
            nxt = nxt / np.mean(nxt)
            if np.linalg.norm(nxt - params, ord=np.inf) < 1e-6:
                return nxt
            params = nxt
        raise RuntimeError('did not converge')


    def predict_prob(self, model_path, model_meta):
        # Add ops to save and restore all the variables.
        saver = tf.train.import_meta_graph(model_meta)

        with tf.Session() as session:
            # Restore variables from disk.
            saver.restore(session, model_path)
            print("Model restored.")
            graph = tf.get_default_graph()
            test_prediction = graph.get_tensor_by_name('test_prediction/test_prediction:0')
            tf_test_left = graph.get_tensor_by_name('test_left:0')
            tf_test_right = graph.get_tensor_by_name('test_right:0')
            tf_test_queries = graph.get_tensor_by_name('test_queries:0')

            test_data_left, test_data_right, test_labels_new, test_data_queries,\
            qPmatDict = Utilities.transform_pairwise(self.test_dataset, self.test_labels,
                                                                                        self.test_queries,
                                                                                        prob=True, eval=True)

            vsteps, test_left_batches, test_right_batches, test_label_batches, test_query_batches = NN.batchData_pairwise(
                data_left=test_data_left,
                data_right=test_data_right,
                labels=test_labels_new,
                queries=test_data_queries,
                batch_size=NNConfig.batch_size)
            predictions = np.zeros(len(test_labels_new))
            idx = 0
            for step in range(0, vsteps):
                batch_predictions = session.run(test_prediction,
                                               feed_dict={tf_test_left: test_left_batches[step],
                                                          tf_test_right: test_right_batches[step],
                                                          tf_test_queries: test_query_batches[step]})
                for i in range(0, len(batch_predictions)):
                    predictions[idx] = batch_predictions[i][0]
                    idx += 1


            lblDict = self.getLblDict()

            for q, v in qPmatDict.items():
                #q query
                #v pairwise comparision of docs in q
                #v[1]: docs in q
                #v[0]: pairwise dict
                print(v[1])
                print("len v: {}".format(len(v[1])))
                print(len(predictions))
                _pmat = np.zeros((len(v[1]), len(v[1])))
                for i, j in v[0].items():
                    _pmat[j[0]][j[1]] = predictions[i]

                # Estimating Bradley-Terry model parameters.
                params = self.mle(_pmat)

                # Ranking (best to worst).
                ranking = np.argsort(params)[::-1]
                r = []
                urlLabels = lblDict[q]
                for urlr in ranking:
                    url = v[1][urlr]
                    if url not in urlLabels:
                        print(url)
                        pass
                    else:
                        print("url2:{}".format(url))
                        r.append(urlLabels[url])

                print('NDCG@{}: {}'.format(10, ndcg_at_k(r, 10)))
                print('NDCG@{}: {}'.format(20, ndcg_at_k(r, 20)))
                print('NDCG@{}: {}'.format(30, ndcg_at_k(r, 30)))




if __name__ == '__main__':
    parser = ArgumentParser(description='Required arguments')
    parser.add_argument('-m', '--modelpath', help='model path', required=True)
    parser.add_argument('-meta', '--metapath', help='model path', required=True)
    args = parser.parse_args()
    eval = Evaluation(pretrained=True)
    #predictions = eval.predict_pointwise(model_path=args.modelpath)
    eval.predict_prob(model_path=args.modelpath, model_meta=args.metapath)
    #eval.computeNDCGforPointwise(predictions)











