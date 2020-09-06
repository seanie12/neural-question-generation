from model import Seq2seq
import os
from data_utils import START_TOKEN, END_ID, get_loader, UNK_ID, outputids2words
import torch
import torch.nn.functional as F
import config
import pickle


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
    def __init__(self, model_path, output_dir):
        with open(config.word2idx_file, "rb") as f:
            word2idx = pickle.load(f)

        self.output_dir = output_dir
        self.test_data = open(config.test_trg_file, "r").readlines()
        self.data_loader = get_loader(config.test_src_file,
                                      config.test_trg_file,
                                      word2idx,
                                      batch_size=1,
                                      use_tag=True,
                                      shuffle=False)

        self.tok2idx = word2idx
        self.idx2tok = {idx: tok for tok, idx in self.tok2idx.items()}
        self.model = Seq2seq()
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.moddel = self.model.to(config.device)
        self.pred_dir = os.path.join(output_dir, "/generated.txt")
        self.golden_dir = os.path.join(output_dir, "/golden.txt")
        self.src_file = os.path.join(output_dir, "src.txt")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # dummy file for evaluation
        with open(self.src_file, "w") as f:
            for i in range(len(self.data_loader)):
                f.write(i+"\n")

    @staticmethod
    def sort_hypotheses(hypotheses):
        return sorted(hypotheses, key=lambda h: h.avg_log_prob, reverse=True)

    def decode(self):
        pred_fw = open(self.pred_dir, "w")
        golden_fw = open(self.golden_dir, "w")
        for i, eval_data in enumerate(self.data_loader):
            src_seq, ext_src_seq, _, \
                _, tag_seq, oov_lst = eval_data

            best_question = self.beam_search(src_seq, ext_src_seq, tag_seq)
            # discard START  token
            output_indices = [int(idx) for idx in best_question.tokens[1:-1]]
            decoded_words = outputids2words(
                output_indices, self.idx2tok, oov_lst[0])
            try:
                fst_stop_idx = decoded_words.index(END_ID)
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words
            decoded_words = " ".join(decoded_words)
            golden_question = self.test_data[i]
            print("write {}th question\r".format(i))
            pred_fw.write(decoded_words + "\n")
            golden_fw.write(golden_question)

        pred_fw.close()
        golden_fw.close()

    def beam_search(self, src_seq, ext_src_seq, tag_seq):
        enc_mask = torch.sign(src_seq)
        src_len = torch.sum(enc_mask, 1)
        prev_context = torch.zeros(1, 1, 2 * config.hidden_size)

        if config.use_gpu:
            src_seq = src_seq.to(config.device)
            ext_src_seq = ext_src_seq.to(config.device)
            src_len = src_len.to(config.device)
            enc_mask = enc_mask.to(config.device)
            prev_context = prev_context.to(config.device)
            tag_seq = tag_seq.to(config.device)
        # forward encoder
        enc_outputs, enc_states = self.model.encoder(src_seq, src_len, tag_seq)
        h, c = enc_states  # [2, b, d] but b = 1
        hypotheses = [Hypothesis(tokens=[self.tok2idx[START_TOKEN]],
                                 log_probs=[0.0],
                                 state=(h[:, 0, :], c[:, 0, :]),
                                 context=prev_context[0]) for _ in range(config.beam_size)]
        # tile enc_outputs, enc_mask for beam search
        ext_src_seq = ext_src_seq.repeat(config.beam_size, 1)
        enc_outputs = enc_outputs.repeat(config.beam_size, 1, 1)
        enc_features = self.model.decoder.get_encoder_features(enc_outputs)
        enc_mask = enc_mask.repeat(config.beam_size, 1)
        num_steps = 0
        results = []
        while num_steps < config.max_decode_step and len(results) < config.beam_size:
            latest_tokens = [h.latest_token for h in hypotheses]
            latest_tokens = [idx if idx < len(
                self.tok2idx) else UNK_ID for idx in latest_tokens]
            prev_y = torch.tensor(latest_tokens, dtype=torch.long).view(-1)

            if config.use_gpu:
                prev_y = prev_y.to(config.device)

            # make batch of which size is beam size
            all_state_h = []
            all_state_c = []
            all_context = []
            for h in hypotheses:
                state_h, state_c = h.state  # [num_layers, d]
                all_state_h.append(state_h)
                all_state_c.append(state_c)
                all_context.append(h.context)

            prev_h = torch.stack(all_state_h, dim=1)  # [num_layers, beam, d]
            prev_c = torch.stack(all_state_c, dim=1)  # [num_layers, beam, d]
            prev_context = torch.stack(all_context, dim=0)
            prev_states = (prev_h, prev_c)
            # [beam_size, |V|]
            logits, states, context_vector = self.model.decoder.decode(prev_y, ext_src_seq,
                                                                       prev_states, prev_context,
                                                                       enc_features, enc_mask)
            h_state, c_state = states
            log_probs = F.log_softmax(logits, dim=1)
            top_k_log_probs, top_k_ids \
                = torch.topk(log_probs, config.beam_size * 2, dim=-1)

            all_hypotheses = []
            num_orig_hypotheses = 1 if num_steps == 0 else len(hypotheses)
            for i in range(num_orig_hypotheses):
                h = hypotheses[i]
                state_i = (h_state[:, i, :], c_state[:, i, :])
                context_i = context_vector[i]
                for j in range(config.beam_size * 2):
                    new_h = h.extend(token=top_k_ids[i][j].item(),
                                     log_prob=top_k_log_probs[i][j].item(),
                                     state=state_i,
                                     context=context_i)
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
