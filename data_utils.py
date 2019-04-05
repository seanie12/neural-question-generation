import torch.utils.data as data
import torch
import config
import pickle
import numpy as np

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
START_TOKEN = "<SOS>"
END_TOKEN = "<EOS>"


class SQuadDataset(data.Dataset):
    def __init__(self, src_file, trg_file, max_length, src_word2idx, trg_word2idx, debug=False):
        self.src = open(src_file, "r", encoding="utf-8").readlines()
        self.trg = open(trg_file, "r", encoding="utf-8").readlines()
        assert len(self.src) == len(self.trg), "the number of source sequence and target sequence must be the same"

        self.max_length = max_length
        self.src_word2idx = src_word2idx
        self.trg_word2idx = trg_word2idx
        self.num_seqs = len(self.src)

        if debug:
            self.src = self.src[:100]
            self.trg = self.trg[:100]
            self.num_seqs = 100

    def __getitem__(self, index):
        src_seq = self.src[index]
        trg_seq = self.trg[index]
        src_seq = self.preprocess(src_seq, self.src_word2idx)
        trg_seq = self.preprocess(trg_seq, self.trg_word2idx)
        return src_seq, trg_seq

    def preprocess(self, sequence, word2idx):
        tokens = sequence.split()
        seq = list()
        seq.append(word2idx[START_TOKEN])
        seq += [word2idx[token] if token in word2idx else word2idx[UNK_TOKEN] for token in tokens]
        seq.append(word2idx[END_TOKEN])
        seq = torch.Tensor(seq)

        return seq


def collate_fn(data):
    def merge(sequences):
        lengths = [len(sequence) for sequence in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    data.sort(key=lambda x: len(x[0]), reverse=True)
    src_seqs, trg_seqs = zip(*data)
    src_seqs, src_len = merge(src_seqs)
    trg_seqs, trg_len = merge(trg_seqs)
    return src_seqs, src_len, trg_seqs, trg_len


def get_loader(src_file, trg_file, src_word2idx, trg_word2idx, batch_size, debug, shuffle=True):
    dataset = SQuadDataset(src_file, trg_file, config.max_len,
                           src_word2idx, trg_word2idx, debug)

    dataloader = data.DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 collate_fn=collate_fn)

    return dataloader


def make_vocab(input_file, output_file, max_vocab_size):
    word2idx = dict()
    word2idx[PAD_TOKEN] = 0
    word2idx[UNK_TOKEN] = 1
    word2idx[START_TOKEN] = 2
    word2idx[END_TOKEN] = 3
    counter = dict()
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.split()
            for token in tokens:
                if token in counter:
                    counter[token] += 1
                else:
                    counter[token] = 1

    sorted_vocab = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)
    for i, (word, _) in enumerate(sorted_vocab, start=4):
        if i == max_vocab_size:
            break
        word2idx[word] = i
    with open(output_file, "wb") as f:
        pickle.dump(word2idx, f)

    return word2idx


def make_embedding(embedding_file, output_file, word2idx):
    word2embedding = dict()
    with open(embedding_file, "r", encoding="utf-8") as f:
        for line in f:
            word_vec = line.split(" ")
            word = word_vec[0]
            vec = np.array(word_vec[1:], dtype=np.float32)
            word2embedding[word] = vec
    embedding = np.zeros((len(word2idx), 300), dtype=np.float32)
    for word, vec in word2embedding.items():
        try:
            idx = word2idx[word]
            embedding[idx] = word2embedding[word]
        except KeyError:
            continue
    with open(output_file, "wb") as f:
        pickle.dump(embedding, f)
    return embedding
