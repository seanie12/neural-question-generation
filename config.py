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

model_path = "./save/c2q/train_502103227/20_2.83"
train = True
device = "cuda:1"
use_gpu = True
debug = False
vocab_size = 30522

num_epochs = 20
max_length = 400
max_seq_len = 400
max_query_len = 64
num_layers = 2
hidden_size = 300
embedding_size = 768

# QA config
qa_lr = 5e-5
gradient_accumulation_steps = 1
warmup_proportion = 0.1
dual_lambda = 0.1

lr = 0.1
batch_size = 64
dropout = 0.3
max_grad_norm = 5.0


use_tag = True
use_pointer = True
beam_size = 10
min_decode_step = 8
max_decode_step = 40
output_dir = "./result/bert_embeddings_ans"
