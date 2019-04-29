import config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch_scatter import scatter_max
from data_utils import UNK_ID
from pytorch_pretrained_bert import BertForQuestionAnswering

INF = 1e12


class Encoder(nn.Module):
    def __init__(self, embeddings, hidden_size, num_layers, dropout, use_tag):
        super(Encoder, self).__init__()
        vocab_size, embedding_size = embeddings.size()

        self.use_tag = use_tag
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        if use_tag:
            self.tag_embedding = nn.Embedding(3, 3)
            lstm_input_size = embedding_size + 3
        else:
            lstm_input_size = embedding_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)

        if embeddings is not None:
            if "FloatTensor" in embeddings.type():
                self.embedding.from_pretrained(embeddings, freeze=True)
            else:
                self.embedding.weight = embeddings
                self.embedding.requires_grad = False

        self.num_layers = num_layers
        if self.num_layers == 1:
            dropout = 0.0
        self.lstm = nn.LSTM(lstm_input_size, hidden_size, dropout=dropout,
                            num_layers=num_layers, bidirectional=True, batch_first=True)
        self.linear_trans = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)
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

    def forward(self, src_seq, src_len, tag_seq):
        total_length = src_seq.size(1)
        embedded = self.embedding(src_seq)
        if self.use_tag and tag_seq is not None:
            tag_embedded = self.tag_embedding(tag_seq)
            embedded = torch.cat((embedded, tag_embedded), dim=2)
        packed = pack_padded_sequence(embedded, src_len, batch_first=True)
        self.lstm.flatten_parameters()
        outputs, states = self.lstm(packed)  # states : tuple of [4, b, d]
        outputs, _ = pad_packed_sequence(outputs, batch_first=True,
                                         total_length=total_length)  # [b, t, d]
        h, c = states

        # self attention
        zeros = outputs.sum(dim=-1)
        mask = (zeros == 0).byte()
        memories = self.linear_trans(outputs)
        outputs = self.gated_self_attn(outputs, memories, mask)

        _, b, d = h.size()
        h = h.view(2, 2, b, d)  # [n_layers, bi, b, d]
        h = torch.cat((h[:, 0, :, :], h[:, 1, :, :]), dim=-1)

        c = c.view(2, 2, b, d)
        c = torch.cat((c[:, 0, :, :], c[:, 1, :, :]), dim=-1)
        concat_states = (h, c)

        return outputs, concat_states


class Decoder(nn.Module):
    def __init__(self, embeddings, hidden_size, num_layers, dropout):
        super(Decoder, self).__init__()
        vocab_size, embedding_size = embeddings.size()
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        if embeddings is not None:
            self.embedding = nn.Embedding(vocab_size, embedding_size)

            if "FloatTensor" in embeddings.type():
                self.embedding.from_pretrained(embeddings, freeze=True)
            else:
                self.embedding.weight = embeddings
                self.embedding.requires_grad = False

        if num_layers == 1:
            dropout = 0.0
        self.encoder_trans = nn.Linear(hidden_size, hidden_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True,
                            num_layers=num_layers, bidirectional=False, dropout=dropout)
        self.concat_layer = nn.Linear(2 * hidden_size, hidden_size, bias=False)
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

    def forward(self, trg_seq, ext_src_seq, init_states, encoder_outputs):
        # trg_seq : [b,t]
        # init_states : [2,b,d]
        # encoder_outputs : [b,t,d]
        # init_states : a tuple of [2, b, d]

        batch_size, max_len = trg_seq.size()
        zeros = encoder_outputs.sum(dim=-1)
        encoder_mask = (zeros == 0).byte()
        memories = self.get_encoder_features(encoder_outputs)
        logits = []
        prev_states = init_states
        for i in range(max_len):
            y_i = trg_seq[:, i].unsqueeze(1)  # [b, 1]
            embedded = self.embedding(y_i)  # [b, 1, d]
            self.lstm.flatten_parameters()
            output, states = self.lstm(embedded, prev_states)
            # encoder-decoder attention
            context, energy = self.attention(output, memories, encoder_mask)
            concat_input = torch.cat((output, context), dim=2).squeeze(dim=1)
            logit_input = torch.tanh(self.concat_layer(concat_input))
            logit = self.logit_layer(logit_input)  # [b, |V|]

            # maxout pointer network
            if config.use_pointer:
                num_oov = max(torch.max(ext_src_seq - self.vocab_size + 1), 0)
                zeros = torch.zeros((batch_size, num_oov), device=config.device)
                extended_logit = torch.cat([logit, zeros], dim=1)
                out = torch.zeros_like(extended_logit) - INF
                out, _ = scatter_max(energy, ext_src_seq, out=out)
                out = out.masked_fill(out == -INF, 0)
                logit = extended_logit + out
                logit = logit.masked_fill(logit == 0, -INF)

            logits.append(logit)
            # update prev state and context
            prev_states = states

        logits = torch.stack(logits, dim=1)  # [b, t, |V|]

        return logits

    def decode(self, y, ext_x, prev_states, encoder_features, encoder_mask):
        # forward one step lstm
        # y : [b]

        embedded = self.embedding(y.unsqueeze(1))
        output, states = self.lstm(embedded, prev_states)
        context, energy = self.attention(output, encoder_features, encoder_mask)
        concat_input = torch.cat((output, context), dim=2).squeeze(dim=1)
        logit_input = torch.tanh(self.concat_layer(concat_input))
        logit = self.logit_layer(logit_input)  # [b, |V|]

        if config.use_pointer:
            batch_size = y.size(0)
            num_oov = max(torch.max(ext_x - self.vocab_size + 1), 0)
            zeros = torch.zeros((batch_size, num_oov), device=config.device)
            extended_logit = torch.cat([logit, zeros], dim=1)
            out = torch.zeros_like(extended_logit) - INF
            out, _ = scatter_max(energy, ext_x, out=out)
            out = out.masked_fill(out == -INF, 0)
            logit = extended_logit + out
            logit = logit.masked_fill(logit == 0, -INF)
            # forcing UNK prob 0
            # logit[:, UNK_ID] = -INF

        return logit, states


class PointerDecoder(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout):
        super(PointerDecoder, self).__init__()

        self.go_embedding = nn.Parameter(torch.randn(1, hidden_size))
        self.concat_layer = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.attention_layer = nn.Linear(hidden_size, 1, bias=False)
        self.lstm = nn.LSTM(hidden_size, hidden_size, dropout=dropout,
                            batch_first=True, num_layers=num_layers)

    def forward(self, enc_outputs, init_states, start_positions):
        """

        :param enc_outputs: [b,t,d] hidden states of encoder
        :param init_states: tuple of [2, b, d]last hidden and cell states of encoder
        :param start_positions: [b] ground truth for start position
        :return: list of start logits and end logits
        """
        batch_size, nsteps, _ = enc_outputs.size()
        enc_mask = (torch.sum(enc_outputs, dim=-1) == 0).byte()
        # tile go embedding
        inputs = self.go_embedding.unsqueeze(0).repeat([batch_size, 1, 1])
        states = init_states
        logits = []
        for i in range(2):
            self.lstm.flatten_parameters()
            hidden, states = self.lstm(inputs, states)
            logit = self.attention(enc_outputs, enc_mask, hidden)
            logits.append(logit)
            # teacher forcing
            inputs = enc_outputs[torch.arange(batch_size), start_positions]
            inputs = inputs.unsqueeze(dim=1)
        return logits

    def attention(self, memories, mask, dec_hidden):
        nsteps = memories.size(1)
        tiled_hidden = dec_hidden.repeat([1, nsteps, 1])
        concat_input = torch.cat([memories, tiled_hidden], dim=2)
        attn_features = torch.tanh(self.concat_layer(concat_input))
        energies = self.attention_layer(attn_features).squeeze(2)  # [b,t,1] -> [b,t]
        logit = energies.masked_fill(mask, -1e12)  # [b, t]
        return logit


class AnswerSelector(nn.Module):
    def __init__(self, embedding=None, model_path=None):
        super(AnswerSelector, self).__init__()
        self.encoder = Encoder(embedding,
                               config.hidden_size,
                               config.num_layers,
                               config.dropout,
                               use_tag=False)
        self.decoder = PointerDecoder(2 * config.hidden_size,
                                      config.num_layers,
                                      config.dropout)
        if model_path is not None:
            ckpt = torch.load(model_path)
            self.encoder.load_state_dict(ckpt["encoder_state_dict"])
            self.decoder.load_state_dict(ckpt["decoder_state_dict"])

    def forward(self, src_seqs, src_len, start_positions):
        tag_seq = None
        enc_outputs, enc_states = self.encoder(src_seqs, src_len, tag_seq)
        logits = self.decoder(enc_outputs, enc_states, start_positions)

        return logits


class Seq2seq(nn.Module):
    def __init__(self, embedding=None, use_tag=False, model_path=None):
        super(Seq2seq, self).__init__()
        encoder = Encoder(embedding,
                          config.hidden_size,
                          config.num_layers,
                          config.dropout,
                          use_tag)
        decoder = Decoder(embedding,
                          2 * config.hidden_size,
                          config.num_layers,
                          config.dropout)

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

    def eval_mode(self):
        self.encoder = self.encoder.eval()
        self.decoder = self.decoder.eval()

    def train_mode(self):
        self.encoder = self.encoder.train()
        self.decoder = self.decoder.train()


class DualNet(nn.Module):
    def __init__(self, c2q_model_path, c2a_model_path):
        super(DualNet, self).__init__()

        self.qa_model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")
        embedding = self.qa_model.bert.embeddings.word_embeddings.weight

        self.ca2q_model = Seq2seq(embedding, use_tag=True)
        self.c_encoder = Encoder(embedding, config.hidden_size,
                                 config.num_layers, config.dropout,
                                 use_tag=False)
        self.c2q_decoder = Decoder(embedding, 2 * config.hidden_size,
                                   config.num_layers, config.dropout)
        self.c2a_decoder = PointerDecoder(2 * config.hidden_size,
                                          config.num_layers,
                                          config.dropout)

    def forward(self, batch_data):
        # sorting for using packed_sequence and padded_pack_sequence
        c_ids, c_lens, tag_ids, q_ids, \
        input_ids, input_mask, segment_ids, \
        start_positions, end_positions, \
        noq_start_positions, noq_end_positions = batch_data

        # sorting for using packed_padded_sequence and pad_packed_sequence
        c_lens, idx = torch.sort(c_lens, descending=True)
        c_ids = c_ids[idx]
        tag_ids = tag_ids[idx]
        q_ids = q_ids[idx]
        input_ids = input_ids[idx]
        input_mask = input_mask[idx]
        segment_ids = segment_ids[idx]
        start_positions = start_positions[idx]
        end_positions = end_positions[idx]
        noq_start_positions = noq_start_positions[idx]
        noq_end_positions = noq_end_positions[idx]

        # QA loss
        qa_loss = self.qa_model(input_ids, segment_ids, input_mask, start_positions, end_positions)

        # QG without answer loss
        enc_outputs, enc_states = self.c_encoder(c_ids, c_lens, None)
        sos_q_ids = q_ids[:, :-1]
        eos_q_ids = q_ids[:, 1:]
        q_logits = self.c2q_decoder(sos_q_ids, c_ids, enc_states, enc_outputs)
        batch_size, nsteps, _ = q_logits.size()
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        preds = q_logits.view(batch_size * nsteps, -1)
        targets = eos_q_ids.contiguous().view(-1)
        c2q_loss = criterion(preds, targets)

        # QG with answer loss
        enc_outputs, enc_states = self.ca2q_model.encoder(c_ids, c_lens, tag_ids)
        q_logits = self.ca2q_model.decoder(sos_q_ids, c_ids, enc_states, enc_outputs)
        preds = q_logits.view(batch_size * nsteps, -1)
        ca2q_loss = criterion(preds, targets)

        # answer span without question
        enc_outputs, states = self.c_encoder(c_ids, c_lens, None)
        logits = self.c2a_decoder(enc_outputs, states, start_positions)
        start_logits, end_logits = logits

        ignored_index = start_logits.size(1)
        start_logits.clamp_(0, ignored_index)
        end_logits.clamp_(0, ignored_index)
        criterion = nn.CrossEntropyLoss(ignore_index=ignored_index)

        start_loss = criterion(start_logits, noq_start_positions)
        end_loss = criterion(end_logits, noq_end_positions)
        c2a_loss = (start_loss + end_loss) / 2

        # regularization loss
        reg_loss = (qa_loss + c2q_loss - ca2q_loss - c2a_loss) ** 2

        qa_loss = (qa_loss + config.dual_lambda * reg_loss)
        c2q_loss = (c2q_loss + config.dual_lambda * reg_loss)
        ca2q_loss = (ca2q_loss + config.dual_lambda * reg_loss)
        c2a_loss = (c2a_loss + config.dual_lambda * reg_loss)
        return qa_loss, c2q_loss, ca2q_loss, c2a_loss
