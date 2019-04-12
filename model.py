import config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from data_utils import UNK_ID


class Encoder(nn.Module):
    def __init__(self, embeddings, vocab_size, embedding_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        if embeddings is not None:
            self.embedding = nn.Embedding(vocab_size, embedding_size). \
                from_pretrained(embeddings, freeze=config.freeze_embedding)

        self.num_layers = num_layers
        if self.num_layers == 1:
            dropout = 0.0
        self.lstm = nn.LSTM(embedding_size, hidden_size, dropout=dropout,
                            num_layers=num_layers, bidirectional=True, batch_first=True)
        self.linear_trans = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.update_layer = nn.Linear(4 * hidden_size, 2 * hidden_size, bias=False)
        self.gate = nn.Linear(4 * hidden_size, 2 * hidden_size, bias=False)

    def gated_self_attn(self, queries, memories, mask):
        # queries: [b,t,d]
        # memories: [b,t,d]
        # mask: [b,t]
        energies = torch.matmul(queries, memories.transpose(1, 2))  # [b, t, t]
        energies = energies.masked_fill(mask.unsqueeze(1), value=-1e12)
        scores = F.softmax(energies, dim=2)
        context = torch.matmul(scores, queries)
        inputs = torch.cat([queries, context], dim=2)
        f_t = torch.tanh(self.update_layer(inputs))
        g_t = torch.sigmoid(self.gate(inputs))
        updated_output = g_t * f_t + (1 - g_t) * queries

        return updated_output

    def forward(self, src_seq, src_len):
        embedded = self.embedding(src_seq)

        packed = pack_padded_sequence(embedded, src_len, batch_first=True)
        outputs, states = self.lstm(packed)  # states : tuple of [4, b, d]
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)  # [b, t, d]
        h, c = states

        # self attention
        mask = (src_seq == 0).byte()
        memories = self.linear_trans(outputs)
        outputs = self.gated_self_attn(outputs, memories, mask)

        if self.num_layers == 2:
            _, b, d = h.size()
            h = h.view(2, 2, b, d)  # [n_layers, bi, b, d]
            h = torch.cat((h[:, 0, :, :], h[:, 1, :, :]), dim=-1)

            c = c.view(2, 2, b, d)
            c = torch.cat((c[:, 0, :, :], c[:, 1, :, :]), dim=-1)
            concat_states = (h, c)
        else:
            h = torch.cat([h[0], h[1]], dim=1).unsqueeze(0)  # [1, b, 2d]
            c = torch.cat([c[0], c[1]], dim=1).unsqueeze(0)  # [1, b, 2d]
            concat_states = (h, c)

        return outputs, concat_states


class Decoder(nn.Module):
    def __init__(self, embeddings, vocab_size, embedding_size, hidden_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        if embeddings is not None:
            self.embedding = nn.Embedding(vocab_size, embedding_size). \
                from_pretrained(embeddings, freeze=config.freeze_embedding)

        if num_layers == 1:
            dropout = 0.0
        self.encoder_trans = nn.Linear(hidden_size, hidden_size)
        self.reduce_layer = nn.Linear(embedding_size + hidden_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True,
                            num_layers=num_layers, bidirectional=False, dropout=dropout)
        self.concat_layer = nn.Linear(2 * hidden_size, hidden_size)
        self.logit_layer = nn.Linear(hidden_size, vocab_size)

    @staticmethod
    def attention(query, memories, mask):
        # query : [b, 1, d]
        energy = torch.matmul(query, memories.transpose(1, 2))  # [b, 1, t]
        energy = energy.squeeze(1).masked_fill(mask, value=-1e12)
        attn_dist = F.softmax(energy, dim=1).unsqueeze(dim=1)  # [b, 1, t]
        context_vector = torch.matmul(attn_dist, memories)  # [b, 1, d]

        return context_vector, energy

    def get_encoder_features(self, encoder_outputs):
        return self.encoder_trans(encoder_outputs)

    def forward(self, trg_seq, ext_src_seq, init_states, encoder_outputs, encoder_mask):
        # trg_seq : [b,t]
        # init_states : [2,b,d]
        # encoder_outputs : [b,t,d]
        # init_states : a tuple of [2, b, d]
        num_oov = torch.max(ext_src_seq - self.vocab_size + 1)
        batch_size, max_len = trg_seq.size()
        hidden_size = encoder_outputs.size(-1)
        memories = self.get_encoder_features(encoder_outputs)
        logits = []
        prev_states = init_states
        prev_context = torch.zeros((batch_size, 1, hidden_size), device=config.device)
        for i in range(max_len):
            y_i = trg_seq[:, i].unsqueeze(1)  # [b, 1]
            embedded = self.embedding(y_i)  # [b, 1, d]
            lstm_inputs = self.reduce_layer(torch.cat([embedded, prev_context], dim=2))
            output, states = self.lstm(lstm_inputs, prev_states)
            # encoder-decoder attention
            context, energy = self.attention(output, memories, encoder_mask)
            concat_input = torch.cat((output, context), dim=2).squeeze(dim=1)
            logit_input = torch.tanh(self.concat_layer(concat_input))
            logit = self.logit_layer(logit_input)  # [b, |V|]

            # pointer network
            if config.use_pointer:
                zeros = torch.zeros((batch_size, num_oov), device=config.device)
                extended_logit = torch.cat([logit, zeros], dim=1)
                logit = extended_logit.scatter_add(1, ext_src_seq, energy)
                logit = logit.masked_fill(logit == 0, value=-1e12)

            logits.append(logit)
            # update prev state and context
            prev_states = states
            prev_context = context

        logits = torch.stack(logits, dim=1)  # [b, t, |V|]

        return logits

    def decode(self, y, ext_x, prev_states, prev_context, encoder_features, encoder_mask):
        # forward one step lstm
        # y : [b]

        embedded = self.embedding(y.unsqueeze(1))
        lstm_inputs = self.reduce_layer(torch.cat([embedded, prev_context], dim=2))
        output, states = self.lstm(lstm_inputs, prev_states)
        context, energy = self.attention(output, encoder_features, encoder_mask)
        concat_input = torch.cat((output, context), dim=2).squeeze(dim=1)
        logit_input = torch.tanh(self.concat_layer(concat_input))
        logit = self.logit_layer(logit_input)  # [b, |V|]

        if config.use_pointer:
            batch_size = y.size(0)
            num_oov = max(torch.max(ext_x - self.vocab_size + 1), 0)
            zeros = torch.zeros((batch_size, num_oov), device=config.device)
            extended_logit = torch.cat([logit, zeros], dim=1)
            logit = extended_logit.scatter_add(1, ext_x, energy)
            logit = logit.masked_fill(logit == 0, value=-1e12)
            # forcing UNK prob 0
            logit[:, UNK_ID] = -1e12

        return logit, states, context


class Seq2seq(nn.Module):
    def __init__(self, enc_embedding=None, dec_embedding=None, is_eval=False, model_path=None):
        super(Seq2seq, self).__init__()
        encoder = Encoder(enc_embedding, config.enc_vocab_size,
                          config.embedding_size, config.hidden_size,
                          config.num_layers,
                          config.dropout)
        decoder = Decoder(dec_embedding, config.dec_vocab_size,
                          config.embedding_size, 2 * config.hidden_size,
                          config.num_layers,
                          config.dropout)

        if config.use_gpu and torch.cuda.is_available():
            device = torch.device(config.device)
            encoder = encoder.to(device)
            decoder = decoder.to(device)

        self.encoder = encoder
        self.decoder = decoder

        if is_eval:
            self.eval_mode()

        if model_path is not None:
            ckpt = torch.load(model_path)
            self.encoder.load_state_dict(ckpt["encoder_state_dict"])
            self.decoder.load_state_dict(ckpt["decoder_state_dict"])

    def eval_mode(self):
        self.encoder = self.encoder.eval()
        self.decoder = self.decoder.eval()

    def train_mode(self):
        self.encoder = self.encoder.train()
        self.decoder = self.decoder.train()
