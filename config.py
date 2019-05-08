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

qa_path = "./save/qa/train_507163217/bert_2_1.103"
ca2q_path = "./save/c2q/best_with_ans/20_2.83"
c2q_path = "./save/c2q/qg_no_ans/16_3.0"
c2a_path = "./save/c2a/selector/2_2.43"
# model_path = "./save/dual/train_506144932/1_2.890"
model_path = "./save/dual/train_507200353/2_2.924"
train = False
device = "cuda:2"
use_gpu = True
debug = False
vocab_size = 30522

num_epochs = 20
max_length = 400
max_seq_len = 384
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
batch_size = 8
dropout = 0.3
max_grad_norm = 5.0

use_tag = True
use_pointer = True
beam_size = 10
min_decode_step = 8
max_decode_step = 40
output_dir = "./result/dual_learning1"
