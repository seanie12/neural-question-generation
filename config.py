# train file
train_src_file = "./data/src-train.txt"
train_trg_file = "./data/tgt-train.txt"
# dev file
dev_src_file = "./data/src-dev.txt"
dev_trg_file = "./data/tgt-dev.txt"
# test file
test_src_file = "./data/src-test.txt"
test_src_file = "./data/tgt-test.txt"
# embedding and dictionary file
src_embedding = "./data/src_embedding.pkl"
trg_embedding = "./data/trg_embedding.pkl"
src_word2idx_file = "./data/src_word2idx.pkl"
trg_word2idx_file = "./data/trg_word2idx.pkl"

device = "cuda:1"
use_gpu = True
debug = False
enc_vocab_size = 45000
dec_vocab_size = 28000

num_epochs = 200
max_len = 120
hidden_size = 600
embedding_size = 300
lr = 1.0
batch_size = 64
dropout = 0.3
max_grad_norm = 5.0

beam_size = 3
min_decode_step = 15