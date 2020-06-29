#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np

from tctk.module.encoder import SelfAttentiveEncoder, BiLSTMEncoder

# ==================================================


class RNNSAClassificationModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        class_num: int,
        pad_id: int,
        emb_size: int = 100,
        hidden_dim: int = 100,
        n_layers: int = 2,
        attn_unit: int = 100,
        attn_hops: int = 2,
        nfc: int = 100,
        dropout: float = 0.5,
    ):
        super(RNNSAClassificationModel, self).__init__()

        self.encoder = SelfAttentiveEncoder(
            vocab_size,
            emb_size,
            hidden_dim,
            n_layers,
            attn_unit,
            attn_hops,
            dropout,
            pad_id,
        )
        self.fc = nn.Linear(hidden_dim * 2 * attn_hops, nfc)

        self.drop = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        self.pred = nn.Linear(nfc, class_num)
        self.softmax = torch.nn.Softmax(dim=-1)

    def init_weights(self, init_range=0.1):
        self.fc.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.fill_(0)
        self.pred.weight.data.uniform_(-init_range, init_range)
        self.pred.bias.data.fill_(0)

    def forward(self, input_ids: torch.LongTensor):
        outp, self.attention = self.encoder(input_ids)
        outp = outp.view(outp.size(0), -1)
        fc = self.tanh(self.fc(self.drop(outp)))
        logits = self.pred(self.drop(fc))
        if type(self.encoder) == BiLSTMEncoder:
            self.attention = None
        return logits

    def predict(self, input_ids: torch.LongTensor, threshold: float = None):
        """
        input_ids: list of input token sequence
        threshold: threshold probability, below threshold returns -1
        """
        logits = self.forward(input_ids).detach().cpu()
        logits = self.softmax(logits).numpy()
        pred = logits.argmax(axis=-1)

        if threshold:
            probs = logits.max(axis=-1)
            pred = [p if pr >= threshold else -1 for p, pr in zip(pred, probs)]
            pred = np.array(pred)
        return pred, logits

    def init_hidden(self, bsz):
        return self.encoder.init_hidden(bsz)

    def encode(self, inp, hidden):
        return self.encoder(inp, hidden)[0]
