import torch
import numpy as np
import random

from torch.utils.data import Dataset


class TextClassificationDataset(Dataset):
    def __init__(self, tok, sents: list, labels: list, max_len: int, do_corruption: bool = True):
        assert len(sents) == len(labels)
        self.x = sents
        self.y = labels
        self.pad_id = tok.token_to_id(tok.pad)
        self.tok = tok
        self.max_len = max_len
        self.do_corruption = do_corruption

    def __getitem__(self, item):
        x, y = self.x[item], self.y[item]
        x, length = self.tokenize(x)
        return torch.LongTensor(x), length, y

    def __len__(self):
        return len(self.x)

    def tokenize(self, text: str):
        tokens = self.tok.text_to_id(text)
        if self.do_corruption:
            if random.random() <= 0.4:
                tokens = self._corruption(tokens)
        length = min(len(tokens), self.max_len)
        if len(tokens) <= self.max_len:
            tokens += [self.pad_id] * (self.max_len - length)
        else:
            tokens = tokens[:self.max_len]
        return tokens, length

    def _corruption(self, x, k=3):
        """
        :param x: input sequence (numpy array)
        :param k: hyperparameter
        :return: permuted result
        """
        q = [i + np.random.uniform(k + 1) for i in range(len(x))]
        tmp = np.c_[x, q]
        x_noise = tmp[tmp[:, 1].argsort()][:, 0]
        return [int(i) for i in x_noise]
