import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from pytorch_pretrained_bert import BertTokenizer, BertAdam, BertForQuestionAnswering

import config
from data_utils import get_loader, eta, user_friendly_time, progress_bar, time_since
from model import Seq2seq, DualNet, AnswerSelector
from squad_utils import convert_examples_to_features, read_squad_examples


class Trainer(object):
    def __init__(self, model_path=None):
        # load dictionary and embedding file
        with open(config.embedding, "rb") as f:
            embedding = pickle.load(f)
            embedding = torch.Tensor(embedding).to(config.device)
        with open(config.word2idx_file, "rb") as f:
            word2idx = pickle.load(f)

        # train, dev loader
        print("load train data")
        self.train_loader = get_loader(config.train_src_file,
                                       config.train_trg_file,
                                       word2idx,
                                       use_tag=config.use_tag,
                                       batch_size=config.batch_size,
                                       debug=config.debug)
        self.dev_loader = get_loader(config.dev_src_file,
                                     config.dev_trg_file,
                                     word2idx,
                                     use_tag=config.use_tag,
                                     batch_size=128,
                                     debug=config.debug)

        train_dir = os.path.join("./save", "seq2seq")
        self.model_dir = os.path.join(train_dir, "train_%d" % int(time.strftime("%m%d%H%M%S")))
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.model = Seq2seq(embedding, config.use_tag, model_path=model_path)
        params = list(self.model.encoder.parameters()) \
                 + list(self.model.decoder.parameters())

        self.lr = config.lr
        self.optim = optim.SGD(params, self.lr, momentum=0.8)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def save_model(self, loss, epoch):
        state_dict = {
            "epoch": epoch,
            "current_loss": loss,
            "encoder_state_dict": self.model.encoder.state_dict(),
            "decoder_state_dict": self.model.decoder.state_dict()
        }
        loss = round(loss, 2)
        model_save_path = os.path.join(self.model_dir, str(epoch) + "_" + str(loss))
        torch.save(state_dict, model_save_path)

    def train(self):
        batch_num = len(self.train_loader)
        self.model.train_mode()
        best_loss = 1e10
        for epoch in range(1, config.num_epochs + 1):
            print("epoch {}/{} :".format(epoch, config.num_epochs), end="\r")
            start = time.time()
            # halving the learning rate after epoch 8
            if epoch >= 8 and epoch % 2 == 0:
                self.lr *= 0.5
                state_dict = self.optim.state_dict()
                for param_group in state_dict["param_groups"]:
                    param_group["lr"] = self.lr
                self.optim.load_state_dict(state_dict)

            for batch_idx, train_data in enumerate(self.train_loader, start=1):
                batch_loss = self.step(train_data)

                self.optim.zero_grad()
                batch_loss.backward()
                # gradient clipping
                nn.utils.clip_grad_norm_(self.model.parameters(), config.max_grad_norm)
                self.optim.step()
                batch_loss = batch_loss.detach().item()
                msg = "{}/{} {} - ETA : {} - loss : {:.4f}" \
                    .format(batch_idx, batch_num, progress_bar(batch_idx, batch_num),
                            eta(start, batch_idx, batch_num), batch_loss)
                print(msg, end="\r")

            val_loss = self.evaluate(msg)
            if val_loss <= best_loss:
                best_loss = val_loss
                self.save_model(val_loss, epoch)

            print("Epoch {} took {} - final loss : {:.4f} - val loss :{:.4f}"
                  .format(epoch, user_friendly_time(time_since(start)), batch_loss, val_loss))

    def step(self, train_data):
        if config.use_tag:
            src_seq, ext_src_seq, src_len, trg_seq, ext_trg_seq, trg_len, tag_seq, _ = train_data
        else:
            src_seq, ext_src_seq, src_len, trg_seq, ext_trg_seq, trg_len, _ = train_data
            tag_seq = None
        src_len = torch.LongTensor(src_len)

        if config.use_gpu:
            src_seq = src_seq.to(config.device)
            ext_src_seq = ext_src_seq.to(config.device)
            src_len = src_len.to(config.device)
            trg_seq = trg_seq.to(config.device)
            ext_trg_seq = ext_trg_seq.to(config.device)
            if config.use_tag:
                tag_seq = tag_seq.to(config.device)
            else:
                tag_seq = None

        enc_outputs, enc_states = self.model.encoder(src_seq, src_len, tag_seq)
        sos_trg = trg_seq[:, :-1]
        eos_trg = trg_seq[:, 1:]

        if config.use_pointer:
            eos_trg = ext_trg_seq[:, 1:]
        logits = self.model.decoder(sos_trg, ext_src_seq, enc_states, enc_outputs)
        batch_size, nsteps, _ = logits.size()
        preds = logits.view(batch_size * nsteps, -1)
        targets = eos_trg.contiguous().view(-1)
        loss = self.criterion(preds, targets)
        return loss

    def evaluate(self, msg):
        self.model.eval_mode()
        num_val_batches = len(self.dev_loader)
        val_losses = []
        for i, val_data in enumerate(self.dev_loader, start=1):
            with torch.no_grad():
                val_batch_loss = self.step(val_data)
                val_losses.append(val_batch_loss.item())
                msg2 = "{} => Evaluating :{}/{}".format(msg, i, num_val_batches)
                print(msg2, end="\r")
        # go back to train mode
        self.model.train_mode()
        val_loss = np.mean(val_losses)

        return val_loss


class QGTrainer(object):
    def __init__(self):
        # load Bert Tokenizer and pre-trained word embedding
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        embedding = BertForQuestionAnswering.from_pretrained("bert-base-uncased") \
            .bert.embeddings.word_embeddings.weight
        self.model = Seq2seq(embedding, use_tag=False)

        train_dir = os.path.join("./save", "c2q")

        self.train_loader = self.get_data_loader("./squad/train-v1.1.json")
        self.dev_loader = self.get_data_loader("./squad/new_dev-v1.1.json")

        self.model_dir = os.path.join(train_dir, "train_%d" % int(time.strftime("%m%d%H%M%S")))
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        params = list(self.model.encoder.parameters()) \
                 + list(self.model.decoder.parameters())

        self.lr = 1.0
        self.optim = optim.Adam(params)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def get_data_loader(self, file):
        train_examples = read_squad_examples(file, is_training=True, debug=config.debug)
        train_features = convert_examples_to_features(train_examples,
                                                      tokenizer=self.tokenizer,
                                                      max_seq_length=config.max_seq_len,
                                                      max_query_length=config.max_query_len,
                                                      doc_stride=128,
                                                      is_training=True)

        all_c_ids = torch.tensor([f.c_ids for f in train_features], dtype=torch.long)
        all_c_lens = torch.sign(torch.sum(all_c_ids, 1)).long()
        all_q_ids = torch.tensor([f.q_ids for f in train_features], dtype=torch.long)
        all_tag_ids = torch.tensor([f.tag_ids for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_c_ids, all_c_lens, all_tag_ids, all_q_ids)
        sampler = RandomSampler(train_data)
        train_loader = DataLoader(train_data, sampler=sampler, batch_size=config.batch_size)

        return train_loader

    def save_model(self, loss, epoch):
        state_dict = {
            "epoch": epoch,
            "current_loss": loss,
            "encoder_state_dict": self.model.encoder.state_dict(),
            "decoder_state_dict": self.model.decoder.state_dict()
        }
        loss = round(loss, 2)
        model_save_path = os.path.join(self.model_dir, str(epoch) + "_" + str(loss))
        torch.save(state_dict, model_save_path)

    def train(self):
        batch_num = len(self.train_loader)
        self.model.train_mode()
        best_loss = 1e10
        for epoch in range(1, config.num_epochs + 1):
            print("epoch {}/{} :".format(epoch, config.num_epochs), end="\r")
            start = time.time()

            for batch_idx, train_data in enumerate(self.train_loader, start=1):
                batch_loss = self.step(train_data)

                self.optim.zero_grad()
                batch_loss.backward()
                # gradient clipping
                nn.utils.clip_grad_norm_(self.model.parameters(), config.max_grad_norm)
                self.optim.step()
                batch_loss = batch_loss.detach().item()
                msg = "{}/{} {} - ETA : {} - loss : {:.4f}" \
                    .format(batch_idx, batch_num, progress_bar(batch_idx, batch_num),
                            eta(start, batch_idx, batch_num), batch_loss)
                print(msg, end="\r")

            # compute validation loss for every epoch
            val_loss = self.evaluate(msg)
            if val_loss <= best_loss:
                best_loss = val_loss
                self.save_model(val_loss, epoch)

            print("Epoch {} took {} - final loss : {:.4f} - val loss :{:.4f}"
                  .format(epoch, user_friendly_time(time_since(start)), batch_loss, val_loss))

    def step(self, train_data):
        c_ids, c_lens, tag_ids, q_ids = train_data
        # exclude unnecessary PAD tokens of c_ids and q_ids
        max_c_len = torch.max(c_lens)
        c_ids = c_ids[:, :max_c_len]
        q_len = torch.sum(torch.sign(q_ids), 1).long()
        max_q_len = torch.max(q_len)
        q_ids = q_ids[:, :max_q_len]

        # sort data by the length of input seq and allocate tensors to gpu device
        c_lens, idx = torch.sort(c_lens)
        c_ids = c_ids[idx].to(config.device)
        c_lens = c_lens.to(config.device)
        q_ids = q_ids[idx].to(config.device)

        if config.use_tag:
            tag_ids = tag_ids[idx].to(config.device)  # we do not use tag seqs
        else:
            tag_ids = None
        # forward Encoder
        enc_outputs, enc_states = self.model.encoder(c_ids, c_lens, tag_ids)

        sos_trg = q_ids[:, :-1]  # exclude END token
        eos_trg = q_ids[:, 1:]  # exclude START token
        # forward decoder
        logits = self.model.decoder(sos_trg, c_ids, enc_states, enc_outputs)
        # compute loss
        batch_size, nsteps, _ = logits.size()
        preds = logits.view(batch_size * nsteps, -1)
        targets = eos_trg.contiguous().view(-1)
        loss = self.criterion(preds, targets)
        return loss

    def evaluate(self, msg):
        self.model.eval_mode()
        num_val_batches = len(self.dev_loader)
        val_losses = []
        for i, val_data in enumerate(self.dev_loader, start=1):
            with torch.no_grad():
                val_batch_loss = self.step(val_data)
                val_losses.append(val_batch_loss.item())
                msg2 = "{} => Evaluating :{}/{}".format(msg, i, num_val_batches)
                print(msg2, end="\r")
        # go back to train mode
        self.model.train_mode()
        val_loss = np.mean(val_losses)

        return val_loss


class C2ATrainer(object):
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        embedding = BertForQuestionAnswering.from_pretrained("bert-base-uncased") \
            .bert.embeddings.word_embeddings.weight

        # instantiate class and allocate it to gpu device
        self.model = AnswerSelector(embedding).to(config.device)
        train_dir = os.path.join("./save", "c2a")
        self.train_loader = self.get_data_loader("./squad/train-v1.1.json")
        self.dev_loader = self.get_data_loader("./squad/new_dev-v1.1.json")
        self.model_dir = os.path.join(train_dir, "train_%d" % int(time.strftime("%m%d%H%M%S")))
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        params = self.model.parameters()
        self.optim = optim.Adam(params)

    def get_data_loader(self, file):
        train_examples = read_squad_examples(file, is_training=True, debug=config.debug)
        train_features = convert_examples_to_features(train_examples,
                                                      tokenizer=self.tokenizer,
                                                      max_seq_length=config.max_seq_len,
                                                      max_query_length=config.max_query_len,
                                                      doc_stride=128,
                                                      is_training=True)

        all_c_ids = torch.tensor([f.c_ids for f in train_features], dtype=torch.long)
        all_c_lens = torch.sum(torch.sign(all_c_ids), 1).long()
        all_noq_start_positions = torch.tensor([f.noq_start_position for f in train_features], dtype=torch.long)
        all_noq_end_positions = torch.tensor([f.noq_end_position for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_c_ids, all_c_lens, all_noq_start_positions, all_noq_end_positions)

        train_loader = DataLoader(train_data, shuffle=True, batch_size=config.batch_size)

        return train_loader

    def save_model(self, loss, epoch):
        loss = round(loss, 2)
        model_save_path = os.path.join(self.model_dir, str(epoch) + "_" + str(loss))
        state_dict = self.model.state_dict()
        torch.save(state_dict, model_save_path)

    def train(self):
        batch_num = len(self.train_loader)
        best_loss = 1e10
        for epoch in range(1, config.num_epochs + 1):
            print("epoch {}/{} :".format(epoch, config.num_epochs), end="\r")
            start = time.time()

            for batch_idx, train_data in enumerate(self.train_loader, start=1):
                batch_loss = self.step(train_data)

                self.optim.zero_grad()
                batch_loss.backward()
                # gradient clipping
                nn.utils.clip_grad_norm_(self.model.parameters(), config.max_grad_norm)
                self.optim.step()
                batch_loss = batch_loss.detach().item()
                msg = "{}/{} {} - ETA : {} - loss : {:.4f}" \
                    .format(batch_idx, batch_num, progress_bar(batch_idx, batch_num),
                            eta(start, batch_idx, batch_num), batch_loss)
                print(msg, end="\r")

            # compute validation loss for every epoch
            val_loss = self.evaluate(msg)
            if val_loss <= best_loss:
                best_loss = val_loss
                self.save_model(val_loss, epoch)

            print("Epoch {} took {} - final loss : {:.4f} - val loss :{:.4f}"
                  .format(epoch, user_friendly_time(time_since(start)), batch_loss, val_loss))

    def step(self, train_data):
        c_ids, c_lens, start_positions, end_positions = train_data

        # sort data allocate tensors to gpu device
        c_lens, idx = torch.sort(c_lens, descending=True)
        c_ids = c_ids[idx].to(config.device)
        c_lens = c_lens.to(config.device)
        start_positions = start_positions[idx].to(config.device)
        end_positions = end_positions[idx].to(config.device)
        # forward pass
        start_logits, end_logits = self.model(c_ids, c_lens, start_positions)
        # compute loss

        ignored_index = start_logits.size(1)
        start_logits.clamp_(0, ignored_index)
        end_logits.clamp_(0, ignored_index)
        criterion = nn.CrossEntropyLoss(ignore_index=ignored_index)

        start_loss = criterion(start_logits, start_positions)
        end_loss = criterion(end_logits, end_positions)
        loss = (start_loss + end_loss) / 2
        return loss

    def evaluate(self, msg):
        self.model.eval()
        num_val_batches = len(self.dev_loader)
        val_losses = []
        for i, val_data in enumerate(self.dev_loader, start=1):
            with torch.no_grad():
                val_batch_loss = self.step(val_data)
                val_losses.append(val_batch_loss.item())
                msg2 = "{} => Evaluating :{}/{}".format(msg, i, num_val_batches)
                print(msg2, end="\r")
        # go back to train mode
        self.model.train()
        val_loss = np.mean(val_losses)

        return val_loss


class DualTrainer(object):
    def __init__(self, c2q_model_path, c2a_model_path):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = DualNet(c2q_model_path, c2a_model_path)
        self.save_dir = os.path.join("./save", "dual")
        self.num_train = None
        # read data-set and prepare iterator
        self.train_loader = self.get_data_loader("./squad/train-v1.1.json")
        self.dev_loader = self.get_data_loader("./squad/new_dev-v1.1.json")

        num_train_optimization_steps = int(
            self.num_train / config.batch_size / config.gradient_accumulation_steps) * config.num_epochs
        # optimizer
        param_optimizer = list(self.model.qa_model.named_parameters())
        # hack to remove pooler, which is not used
        # thus it produce None grad that break apex
        param_optimizer = [n for n in param_optimizer if "pooler" not in n[0]]
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.qa_opt = BertAdam(optimizer_grouped_parameters,
                               lr=config.qa_lr,
                               warmup=config.warmup_proportion,
                               t_total=num_train_optimization_steps)

        params = list(self.model.c_encoder.parameters()) \
                 + list(self.model.ca2q_model.parameters()) \
                 + list(self.model.c2a_decoder.parameters()) \
                 + list(self.model.c2q_decoder.parameters())
        self.general_opt = optim.Adam(params, lr=config.lr)

        # assign model to device and wrap it with Dataparallel
        torch.cuda.set_device(0)
        self.model.cuda()
        self.model = nn.DataParallel(self.model)

        self.model.train()

    def get_data_loader(self, file):
        train_examples = read_squad_examples(file, is_training=True, debug=config.debug)
        train_features = convert_examples_to_features(train_examples,
                                                      tokenizer=self.tokenizer,
                                                      max_seq_length=config.max_seq_len,
                                                      max_query_length=config.max_query_len,
                                                      doc_stride=128,
                                                      is_training=True)
        self.num_train = len(train_examples)
        all_c_ids = torch.tensor([f.c_ids for f in train_features], dtype=torch.long)
        all_c_lens = torch.sign(torch.sum(all_c_ids, 1)).long()
        all_tag_ids = torch.tensor([f.tag_ids for f in train_features], dtype=torch.long)
        all_q_ids = torch.tensor([f.q_ids for f in train_features], dtype=torch.long)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
        all_noq_start_positions = torch.tensor([f.noq_start_position for f in train_features], dtype=torch.long)
        all_noq_end_positions = torch.tensor([f.noq_end_position for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_c_ids, all_c_lens, all_tag_ids,
                                   all_q_ids, all_input_ids, all_input_mask,
                                   all_segment_ids, all_start_positions, all_end_positions,
                                   all_noq_start_positions, all_noq_end_positions)

        sampler = RandomSampler(train_data)
        train_loader = DataLoader(train_data, sampler=sampler, batch_size=config.batch_size)

        return train_loader

    def save_model(self, accuracy, epoch):
        acc = round(accuracy, 3)
        dir_name = os.path.join(self.save_dir, "{:.3f}_{}".format(acc, epoch))
        # save bert model
        model_to_save = self.model.module.qa_model if hasattr(self.model, "module") else self.model.qa_model
        model_file = os.path.join(dir_name, "pytorch_model.bin")
        config_file = os.path.join(dir_name, "bert_config.json")

        state_dict = model_to_save.state_dict()
        torch.save(state_dict, model_file)
        json_file = model_to_save.config.to_json_string(config_file)
        with open(config_file, "w") as f:
            f.write(json_file)

    def train(self):
        global_step = 1
        device = torch.device("cuda")
        batch_num = len(self.train_loader)
        for epoch in range(1, config.num_epochs + 1):
            start = time.time()
            for batch_idx, batch in enumerate(self.train_loader, start=1):
                batch = tuple(t.to(device) for t in batch)
                qa_loss, c2q_loss, ca2q_loss, c2a_loss = self.model(batch)
                # zero grad
                self.qa_opt.zero_grad()
                self.general_opt.zero_grad()
                # mean() to average across multiple gpu and back-propagation
                qa_loss = qa_loss.mean()
                c2q_loss = c2q_loss.mean()
                ca2q_loss = ca2q_loss.mean()
                c2a_loss = c2a_loss.mean()

                qa_loss.backward(retain_graph=True)
                c2q_loss.backward(retain_graph=True)
                ca2q_loss.backward(retain_graph=True)
                c2a_loss.backward()

                # clip gradient
                nn.utils.clip_grad_norm_(self.model.module.c_encoder.parameters(), config.max_grad_norm)
                nn.utils.clip_grad_norm_(self.model.module.c2q_decoder.parameters(), config.max_grad_norm)
                nn.utils.clip_grad_norm_(self.model.module.c2a_decoder.parameters(), config.max_grad_norm)
                nn.utils.clip_grad_norm_(self.model.module.ca2q_model.parameters(), config.max_grad_norm)

                # update params
                self.qa_opt.step()
                self.general_opt.step()
                global_step += 1
                msg = "{}/{} {} - ETA : {} - qa_loss: {:.2f}, c2q_loss :{:.2f}, ca2q_loss :{:.2f}, c2a_loss:{:.2f}" \
                    .format(batch_idx, batch_num, progress_bar(batch_idx, batch_num),
                            eta(start, batch_idx, batch_num),
                            qa_loss.item(), c2q_loss.item(), ca2q_loss.item(), c2a_loss.item())
                print(msg, end="\r")
            print("----------------------------")
