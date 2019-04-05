device = "cuda:1"
src_embedding = "./data/src_embedding.pkl"
trg_embedding = "./data/trg_embedding.pkl"
src_word2idx_file = "./data/src_word2idx"
trg_word2idx_file = "./data/trg_word2idx"

enc_vocab_size = 45000
dec_vocab_size = 28000

max_len = 120
hidden_size = 600
embedding_size = 300
lr = 1.0
batch_size = 64
dropout = 0.3
