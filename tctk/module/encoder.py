#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn

from torch.autograd import Variable
# ==================================================


class BiLSTMEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_size: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float,
        pad_id: int,
    ):
        super(BiLSTMEncoder, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(vocab_size, emb_size)
        self.bilstm = nn.LSTM(
            emb_size, hidden_dim, n_layers, dropout=dropout, bidirectional=True
        )
        self.n_layers = n_layers
        self.nhid = hidden_dim
        self.pad_id = pad_id

        self.encoder.weight.data[self.pad_id] = 0

    def init_weights(self, init_range=0.1):
        self.encoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, inp):
        hidden = [hid.to(inp.device) for hid in self.init_hidden(inp.size(0))]
        emb = self.drop(self.encoder(inp.transpose(0, 1)))
        outp = self.bilstm(emb, hidden)[0]
        outp = torch.transpose(outp, 0, 1).contiguous()
        return outp, emb

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (
            Variable(weight.new(self.n_layers * 2, bsz, self.nhid).zero_()),
            Variable(weight.new(self.n_layers * 2, bsz, self.nhid).zero_()),
        )


class SelfAttentiveEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_size: int,
        hidden_dim: int,
        n_layers: int,
        attn_unit: int,
        attn_hops: int,
        dropout: float,
        pad_id: int,
    ):
        super(SelfAttentiveEncoder, self).__init__()
        self.bilstm = BiLSTMEncoder(
            vocab_size, emb_size, hidden_dim, n_layers, dropout, pad_id
        )
        self.drop = nn.Dropout(dropout)
        self.ws1 = nn.Linear(hidden_dim * 2, attn_unit, bias=False)
        self.ws2 = nn.Linear(attn_unit, attn_hops, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

        self.attention_hops = attn_hops
        self.pad_id = pad_id

    def init_weights(self, init_range=0.1):
        self.ws1.weight.data.uniform_(-init_range, init_range)
        self.ws2.weight.data.uniform_(-init_range, init_range)

    def forward(self, inp):
        outp = self.bilstm(inp)[0]
        size = outp.size()
        # [bsz, len, nhid]
        compressed_embeddings = outp.view(-1, size[2])
        # [bsz*len, nhid*2]d
        transformed_inp = torch.transpose(inp, 0, 1).contiguous()
        # [bsz, len]
        transformed_inp = transformed_inp.view(size[0], 1, size[1])
        # [bsz, 1, len]
        concatenated_inp = [transformed_inp for _ in range(self.attention_hops)]
        concatenated_inp = torch.cat(concatenated_inp, 1)
        # [bsz, hop, len]

        hbar = self.tanh(self.ws1(self.drop(compressed_embeddings)))
        # [bsz*len, attention-unit]
        alphas = self.ws2(hbar).view(size[0], size[1], -1)
        # [bsz, len, hop]
        alphas = torch.transpose(alphas, 1, 2).contiguous()
        # [bsz, hop, len]
        penalized_alphas = alphas + (-10000 * (concatenated_inp == self.pad_id).float())
        # [bsz, hop, len] + [bsz, hop, len]
        alphas = self.softmax(penalized_alphas.view(-1, size[1]))
        # [bsz*hop, len]
        alphas = alphas.view(size[0], self.attention_hops, size[1])
        # [bsz, hop, len]
        return torch.bmm(alphas, outp), alphas

    def init_hidden(self, bsz):
        return self.bilstm.init_hidden(bsz)
