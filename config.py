# train file
train_src_file = "./data/src-train.txt"
train_trg_file = "./data/tgt-train.txt"
# dev file
dev_src_file = "./data/src-dev.txt"
dev_trg_file = "./data/tgt-dev.txt"
# test file
test_src_file = "./data/src-test.txt"
test_trg_file = "./data/tgt-test.txt"
# embedding and dictionary file
src_embedding = "./data/src_embedding.pkl"
trg_embedding = "./data/trg_embedding.pkl"
src_word2idx_file = "./data/src_word2idx.pkl"
trg_word2idx_file = "./data/trg_word2idx.pkl"

model_path = "./save/seq2seq/train_412102429/19_3.22"
train = False
device = "cuda:1"
use_gpu = True
debug = False
enc_vocab_size = 45000
dec_vocab_size = 45000
freeze_embedding = True

num_epochs = 20
max_len = 120
num_layers = 2
hidden_size = 300
embedding_size = 300
lr = 1.0
batch_size = 64
dropout = 0.3
max_grad_norm = 5.0

use_pointer = True
beam_size = 10
min_decode_step = 8
max_decode_step = 30
output_dir = "./result/pointer"
