import torch.utils.data as data
import torch
import config
import pickle
import numpy as np
import time

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
START_TOKEN = "<SOS>"
END_TOKEN = "<EOS>"

PAD_ID = 0
UNK_ID = 1
START_ID = 2
END_ID = 3


class SQuadDataset(data.Dataset):
    def __init__(self, src_file, trg_file, max_length, src_word2idx, trg_word2idx, debug=False):
        self.src = open(src_file, "r").readlines()
        self.trg = open(trg_file, "r").readlines()
        assert len(self.src) == len(self.trg), \
            "the number of source sequence {}" " and target sequence {} must be the same" \
                .format(len(self.src), len(self.trg))

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

    def __len__(self):
        return self.num_seqs

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


def time_since(t):
    """ Function for time. """
    return time.time() - t


def progress_bar(completed, total, step=5):
    """ Function returning a string progress bar. """
    percent = int((completed / total) * 100)
    bar = '[='
    arrow_reached = False
    for t in range(step, 101, step):
        if arrow_reached:
            bar += ' '
        else:
            if percent // t != 0:
                bar += '='
            else:
                bar = bar[:-1]
                bar += '>'
                arrow_reached = True
    if percent == 100:
        bar = bar[:-1]
        bar += '='
    bar += ']'
    return bar


def user_friendly_time(s):
    """ Display a user friendly time from number of second. """
    s = int(s)
    if s < 60:
        return "{}s".format(s)

    m = s // 60
    s = s % 60
    if m < 60:
        return "{}m {}s".format(m, s)

    h = m // 60
    m = m % 60
    if h < 24:
        return "{}h {}m {}s".format(h, m, s)

    d = h // 24
    h = h % 24
    return "{}d {}h {}m {}s".format(d, h, m, s)


def eta(start, completed, total):
    """ Function returning an ETA. """
    # Computation
    took = time_since(start)
    time_per_step = took / completed
    remaining_steps = total - completed
    remaining_time = time_per_step * remaining_steps

    return user_friendly_time(remaining_time)
