import numpy as np
import tensorflow as tf
from Util.configs import NNConfig
from Util.dataloader import DataLoader
import logging
from argparse import ArgumentParser
class NN:
    def __init__(self):
        logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                            datefmt='%d.%m.%Y %I:%M:%S %p', level=logging.INFO)


        self.log = logging.getLogger("Simple NN")

        self.d_loader = DataLoader()
        self.input_vector_size = self.d_loader.d_handler.get_vocab_size()
        self.output_vector_size = self.d_loader.d_handler.get_vocab_size()
        self.train_dataset, self.train_labels, self.valid_dataset, \
        self.valid_labels, self.test_dataset, self.test_labels = self.d_loader.get_ttv()


    def simpleNN(self, mode="/cpu:0"):

        self.log.info("Create the computational graph..")
        graph = tf.Graph()
        with graph.as_default():
            with tf.device(mode):
                '''
                Input data
                '''
                tf_train_dataset = tf.placeholder(tf.float32, shape=(NNConfig.batch_size, self.input_vector_size))
                tf_train_labels = tf.placeholder(tf.float32, shape=(NNConfig.batch_size, self.output_vector_size))
                tf_valid_dataset = tf.constant(self.valid_dataset)
                tf_test_dataset = tf.constant(self.test_dataset)


                if NNConfig.regularization:
                    beta_regu = tf.placeholder(tf.float32)

                if NNConfig.learning_rate_decay:
                    global_step = tf.Variable(0)


                # Variables
                def init_weights(shape):
                    return tf.Variable(tf.truncated_normal(shape))

                def init_biases(shape):
                    return tf.Variable(tf.zeros(shape))

                w_h = init_weights([self.input_vector_size, NNConfig.num_hidden_nodes])
                b_h = init_biases([NNConfig.num_hidden_nodes])

                w_o = init_weights([NNConfig.num_hidden_nodes, self.output_vector_size])
                b_o = init_biases([self.output_vector_size])

                # Training computation

                def model(dataset, w_h, b_h, w_o, b_o, train_mode):
                    if NNConfig.dropout and train_mode:
                        drop_i = tf.nn.dropout(dataset, NNConfig.dropout_keep_prob_input)
                        h_lay_train = tf.nn.relu(tf.matmul(drop_i, w_h) + b_h)
                        drop_h = tf.nn.dropout(h_lay_train, NNConfig.dropout_keep_prob_hidden)
                        return tf.matmul(drop_h, w_o) + b_o
                    else:
                        h_lay_train = tf.nn.relu(tf.matmul(dataset, w_h) + b_h)
                        return tf.matmul(h_lay_train, w_o) + b_o

                logits = model(tf_train_dataset, w_h, b_h, w_o, b_o, True)
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                                              labels=tf_train_labels))

                if NNConfig.regularization:
                    loss += beta_regu * (tf.nn.l2_loss(w_h) + tf.nn.l2_loss(w_o))
                if NNConfig.learning_rate_decay:
                    learning_rate = tf.train.exponential_decay(NNConfig.learning_rate,
                                                               global_step=global_step,
                                                               decay_steps=NNConfig.decay_steps,
                                                               decay_rate=NNConfig.decay_rate,
                                                               staircase=True)
                    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(
                        loss,global_step=global_step
                    )
                else:
                    optimizer = tf.train.GradientDescentOptimizer(NNConfig.learning_rate).minimize(loss)

                train_prediction = tf.nn.softmax(logits=logits)
                valid_prediction = tf.nn.softmax(model(tf_valid_dataset, w_h, b_h, w_o, b_o, train_mode=False))
                test_prediction = tf.nn.softmax(model(tf_test_dataset, w_h, b_h, w_o, b_o, train_mode=False))


                with tf.name_scope('accuracy'):
                    pre = tf.placeholder("float", shape=[None, self.output_vector_size])
                    lbl = tf.placeholder("float", shape=[None, self.output_vector_size])
                    accuracy = tf.reduce_mean(tf.cast(tf.nn.sigmoid_cross_entropy_with_logits(logits=ls,labels=lbl), "float"))

            self.log.info("running the session..")

            with tf.Session(graph=graph, config=tf.ConfigProto(log_device_placement=True)) as session:
                tf.initialize_all_variables().run()
                self.log.info("Initialized")
                for step in range(NNConfig.num_steps):
                    offset = (step * NNConfig.batch_size) % (self.train_labels.shape[0] - NNConfig.batch_size)
                    batch_data = self.train_dataset[offset:(offset + NNConfig.batch_size), :]
                    batch_labels = self.train_labels[offset:(offset + NNConfig.batch_size), :]

                    feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
                    if NNConfig.regularization:
                        feed_dict[beta_regu] = NNConfig.beta_regu

                    _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
                    if (step % NNConfig.summary_steps == 0):
                        self.log.info("Minibatch loss at step %d: %f" %(step, l))
                        self.log.info("Minibatch accuracy: %.3f%%" % session.run(accuracy, feed_dict={pre: predictions,
                                                                                                      lbl: self.valid_labels}))
                        # self.print_words(predictions, batch_labels)
                        self.log.info('Validation accuracy:  %.3f%%' % session.run(accuracy,
                                                                                 feed_dict={
                                                                                     pre: valid_prediction.eval(),
                                                                                     lbl: self.valid_labels}))
                self.log.info("Test accuracy: %.3%%" % session.run(accuracy, feed_dict={pre: test_prediction.eval(),
                                                                                                lbl: self.test_labels}))


                #self.print_words(test_prediction.eval(), self.test_labels)

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


if __name__ == '__main__':
    parser = ArgumentParser(description='Required arguments')
    parser.add_argument('-m', '--mode', help='computation mode', required=False)
    args = parser.parse_args()
    nn = NN()
    try:
        if args.mode == "gpu":
            nn.simpleNN(mode="/gpu:0")
        else: nn.simpleNN()
        nn.log.info("done..")
    except Exception as e:
        nn.log.exception(e)
        raise











