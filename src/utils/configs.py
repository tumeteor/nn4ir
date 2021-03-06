class NNConfig:
    vocab_size = 100000
    train_ratio = .7
    valid_ratio = .15
    test_ratio = .15
    max_doc_size = 2000
    embedding_dim = 300
    summary_steps = 1000
    num_hidden_nodes = 1024
    num_hidden_layers = 4
    batch_size = 100
    num_steps = 200
    beta_regu = 1e-3
    dropout_keep_prob_input = 1
    dropout_keep_prob_hidden = 0.5
    learning_rate = 0.01
    decay_steps = 1000
    decay_rate = 0.65
    regularization = True
    dropout = True
    learning_rate_decay = False
    sampler = None  # uniform, log_uniform, learned_unigram, fixed_unigram
    candidate_sampling = None  # nce_loss, softmax_loss
    num_sampled = 64  # Number of negative examples to sample.
    train_optimizer = None  # GradientDescent, Adadelta, Adagrad, Momentum, Adam, Ftrl, RMSProp,

    label_size = 2

