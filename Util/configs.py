class DataConfig():
    all_doc_path = "/home/nguyen/nn4ir/data/docs1k/1rev/nn/"
    save_dir_data = "/home/nguyen/nn4ir/data/docs1k/save/data"
    save_dir_model = "/home/nguyen/nn4ir/data/docs1k/save/model"
    qrel_path = "/home/nguyen/nn4ir/data/docs1k/top1k.qu.labels"
    run_path = "/home/nguyen/nn4ir/data/docs1k/top1k.qu.labels"
    qtitle_path = "/home/nguyen/nn4ir/data/docs1k/top1k.qtitles"
    glove_file_path = "/home/nguyen/nn4ir/embeddings/de_wiki/de.bin"
    vocab_size = 100000
    train_ratio = .7
    valid_ratio = .15
    test_ratio = .15
    max_doc_size = 2000

class NNConfig():
    embedding_dim = 300
    summary_steps = 100
    num_hidden_nodes = 1024
    batch_size = 100
    num_steps = 800
    beta_regu = 1e-3
    dropout_keep_prob_input = 1
    dropout_keep_prob_hidden = 0.5
    learning_rate = 0.5
    decay_steps = 1000
    decay_rate = 0.65
    regularization = True
    dropout = True
    learning_rate_decay = True
    sampler = None  # uniform, log_uniform, learned_unigram, fixed_unigram
    candidate_sampling = None  # nce_loss, softmax_loss
    num_sampled = 64  # Number of negative examples to sample.
    train_optimizer = None  # GradientDescent, Adadelta, Adagrad, Momentum, Adam, Ftrl, RMSProp,
