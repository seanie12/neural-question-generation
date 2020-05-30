# train file
train_src_file = "./squad/para-train.txt"
train_trg_file = "./squad/tgt-train.txt"
# dev file
dev_src_file = "./squad/para-dev.txt"
dev_trg_file = "./squad/tgt-dev.txt"
# test file
test_src_file = "./squad/para-test.txt"
test_trg_file = "./squad/tgt-test.txt"
# embedding and dictionary file
embedding = "./data/embedding.pkl"
word2idx_file = "./data/word2idx.pkl"

model_path = "./save/model.pt"

device = "cuda:0"
use_gpu = True
debug = False
vocab_size = 45000
freeze_embedding = True

num_epochs = 20
max_len = 400
num_layers = 2
hidden_size = 300
embedding_size = 300
lr = 0.1
batch_size = 64
dropout = 0.3
max_grad_norm = 5.0

use_pointer = True
beam_size = 10
min_decode_step = 8
max_decode_step = 30
output_dir = "./result/pointer_maxout_ans"
