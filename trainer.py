import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config
from data_utils import get_loader, eta, user_friendly_time, progress_bar, time_since
from model import Seq2seq


class Trainer(object):
    def __init__(self, model_path=None):
        # load dictionary and embedding file
        with open(config.src_embedding, "rb") as f:
            src_embedding = pickle.load(f)
            src_embedding = torch.Tensor(src_embedding).to(config.device)
        with open(config.trg_embedding, "rb") as f:
            trg_embedding = pickle.load(f)
            trg_embedding = torch.Tensor(trg_embedding).to(config.device)
        with open(config.src_word2idx_file, "rb") as f:
            src_word2idx = pickle.load(f)
        with open(config.trg_word2idx_file, "rb") as f:
            trg_word2idx = pickle.load(f)

        # train, dev loader
        print("load train data")
        self.train_loader = get_loader(config.train_src_file,
                                       config.train_trg_file,
                                       src_word2idx,
                                       trg_word2idx,
                                       batch_size=config.batch_size,
                                       debug=config.debug)
        self.dev_loader = get_loader(config.dev_src_file,
                                     config.dev_trg_file,
                                     src_word2idx,
                                     trg_word2idx,
                                     batch_size=512,
                                     debug=config.debug)

        train_dir = os.path.join("./save", "seq2seq")
        self.model_dir = os.path.join(train_dir, "train_%d" % int(time.strftime("%m%d%H%M%S")))
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.model = Seq2seq(src_embedding, trg_embedding, model_path=model_path)
        params = list(self.model.encoder.parameters()) \
                 + list(self.model.decoder.parameters())

        self.lr = config.lr
        self.optim = optim.SGD(params, self.lr)
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
            # halving the learning rate at epoch 8
            if epoch == 8:
                self.lr *= 0.5
            for batch_idx, train_data in enumerate(self.train_loader, start=1):
                src_seq, src_len, trg_seq, trg_len = train_data
                batch_loss = self.step(src_seq, src_len, trg_seq, trg_len)

                self.optim.zero_grad()
                batch_loss.backward()
                # gradient clipping
                nn.utils.clip_grad_norm_(self.model.encoder.parameters(), config.max_grad_norm)
                nn.utils.clip_grad_norm_(self.model.decoder.parameters(), config.max_grad_norm)
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

    def step(self, src_seq, src_len, trg_seq, trg_len):
        src_len = torch.LongTensor(src_len)
        enc_zeros = torch.zeros_like(src_seq)
        enc_mask = torch.ByteTensor(src_seq == enc_zeros)

        if config.use_gpu:
            src_seq = src_seq.to(config.device)
            src_len = src_len.to(config.device)
            trg_seq = trg_seq.to(config.device)
            enc_mask = enc_mask.to(config.device)

        enc_outputs, enc_states = self.model.encoder(src_seq, src_len)
        sos_trg = trg_seq[:, :-1]
        eos_trg = trg_seq[:, 1:]
        logits = self.model.decoder(sos_trg, enc_states, enc_outputs, enc_mask)
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
            src_seq, src_len, trg_seq, trg_len = val_data
            with torch.no_grad():
                val_batch_loss = self.step(src_seq, src_len, trg_seq, trg_len)
                val_losses.append(val_batch_loss.item())
                msg2 = "{} => Evaluating :{}/{}".format(msg, i, num_val_batches)
                print(msg2, end="\r")
        # go back to train mode
        self.model.train_mode()
        val_loss = np.mean(val_losses)

        return val_loss
