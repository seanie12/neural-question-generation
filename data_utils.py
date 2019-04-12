import torch.utils.data as data
import torch
import config
import pickle
import numpy as np
import time
from tqdm import tqdm

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "UNKNOWN"
START_TOKEN = "<s>"
END_TOKEN = "EOS"

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
        src_seq, ext_src_seq, oov_lst = self.context2ids(src_seq, self.src_word2idx)
        trg_seq, ext_trg_seq = self.question2ids(trg_seq, self.trg_word2idx, oov_lst)
        return src_seq, ext_src_seq, trg_seq, ext_trg_seq, oov_lst

    def __len__(self):
        return self.num_seqs

    def context2ids(self, sequence, word2idx):
        ids = list()
        extended_ids = list()
        oov_lst = list()
        ids.append(word2idx[START_TOKEN])
        extended_ids.append(word2idx[START_TOKEN])
        tokens = sequence.strip().split(" ")

        for token in tokens:
            if token in word2idx:
                ids.append(word2idx[token])
                extended_ids.append(word2idx[token])
            else:
                ids.append(word2idx[UNK_TOKEN])
                if token not in oov_lst:
                    oov_lst.append(token)
                extended_ids.append(len(word2idx) + oov_lst.index(token))
        ids.append(word2idx[END_TOKEN])
        extended_ids.append(word2idx[END_TOKEN])

        ids = torch.Tensor(ids)
        extended_ids = torch.Tensor(extended_ids)

        return ids, extended_ids, oov_lst

    def question2ids(self, sequence, word2idx, oov_lst):
        ids = list()
        extended_ids = list()
        ids.append(word2idx[START_TOKEN])
        extended_ids.append(word2idx[START_TOKEN])
        tokens = sequence.strip().split(" ")

        for token in tokens:
            if token in word2idx:
                ids.append(word2idx[token])
                extended_ids.append(word2idx[token])
            else:
                ids.append(word2idx[UNK_TOKEN])
                if token in oov_lst:
                    extended_ids.append(len(word2idx) + oov_lst.index(token))
                else:
                    extended_ids.append(word2idx[UNK_TOKEN])
        ids.append(word2idx[END_TOKEN])
        extended_ids.append(word2idx[END_TOKEN])

        ids = torch.Tensor(ids)
        extended_ids = torch.Tensor(extended_ids)

        return ids, extended_ids


def collate_fn(data):
    def merge(sequences):
        lengths = [len(sequence) for sequence in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    data.sort(key=lambda x: len(x[0]), reverse=True)
    src_seqs, ext_src_seqs, trg_seqs, ext_trg_seqs, oov_lst = zip(*data)

    src_seqs, src_len = merge(src_seqs)
    ext_src_seqs, _ = merge(ext_src_seqs)
    trg_seqs, trg_len = merge(trg_seqs)
    ext_trg_seqs, _ = merge(ext_trg_seqs)

    return src_seqs, ext_src_seqs, src_len, trg_seqs, ext_trg_seqs, trg_len, oov_lst


def get_loader(src_file, trg_file, src_word2idx, trg_word2idx,
               batch_size, debug=False, shuffle=True):
    dataset = SQuadDataset(src_file, trg_file, config.max_len,
                           src_word2idx, trg_word2idx, debug)

    dataloader = data.DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 collate_fn=collate_fn)

    return dataloader


def make_vocab(src_file, trg_file, output_file, max_vocab_size):
    word2idx = dict()
    word2idx[PAD_TOKEN] = 0
    word2idx[UNK_TOKEN] = 1
    word2idx[START_TOKEN] = 2
    word2idx[END_TOKEN] = 3
    counter = dict()
    with open(src_file, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.split()
            for token in tokens:
                if token in counter:
                    counter[token] += 1
                else:
                    counter[token] = 1
    with open(trg_file, "r", encoding="utf-8") as f:
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
    lines = open(embedding_file, "r", encoding="utf-8").readlines()
    for line in tqdm(lines):
        word_vec = line.split(" ")
        word = word_vec[0]
        vec = np.array(word_vec[1:], dtype=np.float32)
        word2embedding[word] = vec
    embedding = np.zeros((len(word2idx), 300), dtype=np.float32)
    num_oov = 0
    for word, idx in word2idx.items():
        if word in word2embedding:
            embedding[idx] = word2embedding[word]
        else:
            embedding[idx] = word2embedding[UNK_TOKEN]
            num_oov += 1
    print(num_oov)
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


def get_embeddings(src_embedding_file, trg_embedding_file):
    with open(src_embedding_file, "rb") as f:
        src_embedding = pickle.load(f)
    with open(trg_embedding_file, "rb") as f:
        trg_embedding = pickle.load(f)

    return src_embedding, trg_embedding


def outputids2words(id_list, idx2word, article_oovs=None):
    """
    :param id_list: list of indices
    :param idx2word: dictionary mapping idx to word
    :param article_oovs: list of oov words
    :return: list of words
    """
    words = []
    for idx in id_list:
        try:
            word = idx2word[idx]
        except KeyError:
            if article_oovs is not None:
                article_oov_idx = idx - len(idx2word)
                try:
                    word = article_oovs[article_oov_idx]
                except IndexError:
                    print("there's no such a word in extended vocab")
            else:
                word = idx2word[UNK_ID]
        words.append(word)

    return words
