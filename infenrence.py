from model import Seq2seq
import os
from data_utils import START_ID, END_ID
import torch
import torch.nn.functional as F
import config


class Hypothesis(object):
    def __init__(self, tokens, log_probs, state, context=None):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.context = context

    def extend(self, token, log_prob, state, context=None):
        h = Hypothesis(tokens=self.tokens + [token],
                       log_probs=self.log_probs + [log_prob],
                       state=state,
                       context=context)
        return h

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.tokens)


class BeamSearcher(object):
    def __init__(self, data_loader, test_file, tok2idx, model_path, output_dir):
        self.test_data = open(test_file, "r").readlines()
        self.data_loader = data_loader
        self.tok2idx = tok2idx
        self.idx2tok = {idx: tok for tok, idx in tok2idx.items()}
        self.model = Seq2seq(model_path=model_path)
        self.pred_dir = os.path.join(output_dir, "generated.txt")
        self.golden_dir = os.paht.join(output_dir, "golden.txt")

    @staticmethod
    def sort_hypotheses(hypotheses):
        return sorted(hypotheses, key=lambda h: h.avg_log_prob, reverse=True)

    def decode(self):
        pred_fw = open(self.pred_dir, "w")
        golden_fw = open(self.golden_dir, "w")
        for i, eval_data in enumerate(self.data_loader, start=1):
            src_seq, src_len, _, _ = eval_data
            best_question = self.beam_search(src_seq, src_len)
            # discard START token
            output_indice = [int(idx) for idx in best_question.tokens[1:]]
            decoded_words = [self.idx2tok(idx) for idx in output_indice]

            try:
                fst_stop_idx = decoded_words.index(END_ID)
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words
            decoded_words = " ".join(decoded_words)
            golden_question = self.test_data[i]
            pred_fw.write(decoded_words + "\n")
            golden_fw.write(golden_question + "\n")

    def beam_search(self, src_seq, src_len):
        zeros = torch.zeros_like(src_seq)
        enc_mask = torch.ByteTensor(src_seq == zeros)

        if config.use_gpu:
            src_seq = src_seq.to(config.device)
            src_len = src_len.to(config.device)
            enc_mask = enc_mask.to(config.device)
        # forward encoder
        enc_outputs, enc_states = self.model.encoder(src_seq, src_len)
        h, c = enc_states  # [2, b, d] but b = 1
        hypotheses = [Hypothesis(tokens=[self.tok2idx[START_ID]],
                                 log_probs=[0.0],
                                 state=(h[:, 0, :], c[:, 0, :]),
                                 context=None)]
        # tile enc_outputs, enc_mask for beam search
        enc_outputs = enc_outputs.repeat(config.beam_size, 1, 1)
        enc_features = self.model.decoder.get_encoder_features(enc_outputs)
        enc_mask = enc_mask.repeat(config.beam_size, 1)
        num_steps = 0
        results = []
        while num_steps < 0 and len(results) < config.beam_size:
            latest_tokens = [h.latest_token for h in hypotheses]

            prev_y = torch.LongTensor(latest_tokens).view(-1)

            if config.use_gpu:
                prev_y = prev_y.to(config.device)

            # make batch of which size is beam size
            all_state_h = []
            all_state_c = []

            for h in hypotheses:
                state_h, state_c = h.state  # [num_layers, d]
                all_state_h.append(state_h)
                all_state_c.append(state_c)
            prev_h = torch.stack(all_state_h, dim=1)  # [num_layers, beam, d]
            prev_c = torch.stack(all_state_c, dim=1)  # [num_layers, beam, d]
            prev_states = (prev_h, prev_c)

            # [beam_size, |V|]
            logits, states = self.model.decoder.decode(prev_y, prev_states,
                                                       enc_features, enc_mask)
            h, c = states
            log_probs = F.log_softmax(logits, dim=1)
            top_k_log_probs, top_k_ids \
                = torch.topk(log_probs, config.beam_size * 2, dim=-1)

            all_hypotheses = []
            num_orig_hypotheses = 1 if num_steps == 0 else len(hypotheses)
            for i in range(num_orig_hypotheses):
                h = hypotheses[i]
                state_i = (h[:, i, :], c[:, i:, :])

                for j in range(config.beam_size * 2):
                    new_h = h.extend(token=top_k_ids[i][j].item(),
                                     log_prob=top_k_log_probs[i][j].item(),
                                     state=state_i)
                    all_hypotheses.append(new_h)

            hypotheses = []
            for h in self.sort_hypotheses(all_hypotheses):
                if h.latest_token == END_ID:
                    if num_steps >= config.min_decode_step:
                        results.append(h)
                else:
                    hypotheses.append(h)

                if len(hypotheses) == config.beam_size or len(results) == config.beam_size:
                    break
            num_steps += 1
        if len(results) == 0:
            results = hypotheses
        h_sorted = self.sort_hypotheses(results)

        return h_sorted[0]
