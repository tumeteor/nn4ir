class DataConfig():
    all_doc_path = None
    save_dir_data = None
    save_dir_model = None
    qrel_path = None
    run_path = None
    vocab_size = 100000
    train_ratio = .7
    valid_ratio = .15
    test_ratio = .15

class NNConfig():
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
