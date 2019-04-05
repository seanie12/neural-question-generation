import config
import torch
import torch.nn as nn
import torch.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class Encoder(nn.Module):
    def __init__(self, embeddings, vocab_size, embedding_size, hidden_size, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size).from_pretrained(embeddings,
                                                                                  freeze=True)
        self.lstm = nn.LSTM(embedding_size, hidden_size, dropout=dropout,
                            num_layers=2, bidirectional=True, batch_first=True)

    def forward(self, src_seq, src_len):
        embedded = self.embedding(src_seq)
        b, t, d = embedded.size()
        packed = pack_padded_sequence(embedded, src_len, batch_first=True)
        outputs, states = self.lstm(packed)  # states : tuple of [4, b, d]
        outputs = pad_packed_sequence(packed, batch_first=True)  # [b, t, d]

        h, c = states
        h = h.view(2, 2, b, d)  # [n_layers, bi, b, d]
        h = torch.cat((h[:, 0, :, :], h[:, 1, :, :]), dim=-1)
        c = c.view(2, 2, b, d)
        c = torch.cat((c[:, 0, :, :], c[:, 1, :, :]), dim=-1)
        concat_states = (h, c)

        return outputs, concat_states


class Decoder(nn.Module):
    def __init__(self, embeddings, vocab_size, embedding_size, hidden_size, dropout):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(embedding_size, vocab_size).from_pretrained(embeddings,
                                                                                  freeze=True)
        self.encoder_trans = nn.Linear(hidden_size, hidden_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True,
                            num_layers=2, bidirectional=False, dropout=dropout)
        self.logit_layer = nn.Linear(2 * hidden_size, vocab_size)

    @staticmethod
    def attention(query, memories, mask):
        # query : [b, 1, d]
        energy = torch.matmul(query, memories.transpose(1, 2))  # [b, 1, t]
        score = energy.squeeze(0).masked_fill(mask, value=-1e12)
        score = F.softmax(score, dim=1).unsqueeze(dim=1)  # [b, 1, t]
        context_vector = torch.matmul(score, memories)  # [b, 1, d]

        return context_vector

    def get_encoder_features(self, encoder_outputs):
        return self.encoder_trans(encoder_outputs)

    def forward(self, trg_seq, init_states, encoder_outputs, encoder_mask):
        # trg_seq : [b,t]
        # encoder_outputs : [b,t,d]
        # init_states : a tuple of [2, b, d]
        max_len = trg_seq.size(1)
        memories = self.get_encoder_features(encoder_outputs)
        logits = []
        prev_states = init_states
        for i in range(max_len):
            y_i = trg_seq[:, i].unsqueeze(1)  # [b, 1]
            embedded = self.embedding(y_i)  # [b, 1, d]
            output, states = self.lstm(embedded, prev_states)
            prev_states = states
            context = self.attention(output, memories, encoder_mask)
            logit_input = torch.cat((output, context), dim=2).squeeze(dim=1)
            logit = self.logit_layer(logit_input)  # [b, |V|]
            logits.append(logit)

        logits = torch.stack(logits, dim=1)  # [b, t, |V|]

        return logits

    def decode(self, y, prev_states, encoder_features, encoder_mask):
        # forward one step lstm
        # y : [b]
        embedded = self.embedding(y.unsqueeze(1))
        output, states = self.lstm(embedded, prev_states)
        context = self.attention(output, encoder_features, encoder_mask)
        logit_input = torch.cat((output, context), dim=2).squeeze(dim=1)
        logit = self.logit_layer(logit_input)  # [b, |V|]

        return logit


class Seq2seq(object):
    def __init__(self, enc_embedding, dec_embedding, is_eval=False, model_path=None):
        super(Seq2seq, self).__init__()
        encoder = Encoder(enc_embedding, config.enc_vocab_size,
                          config.embedding_size, config.hidden_size,
                          config.dropout)
        decoder = Decoder(dec_embedding, config.dec_vocab_size,
                          config.embedding_size, 2 * config.hidden_size,
                          config.dropout)
        if is_eval:
            encoder = encoder.eval()
            decoder = decoder.eval()

        if config.use_gpu and torch.cuda.is_available():
            device = torch.device(config.device)
            encoder = encoder.to(device)
            decoder = decoder.to(device)

        self.encoder = encoder
        self.decoder = decoder

        if model_path is not None:
            ckpt = torch.load(model_path)
            self.encoder.load_state_dict(ckpt["encoder_state_dict"])
            self.decoder.load_state_dict(ckpt["decoder_state_dict"])
