import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import dill
import os

from tctk.tokenizer.sentencepiece import SentencePieceTokenizer
from tctk.model.rnnsa import RNNSAClassificationModel
from tctk.dataset import TextClassificationDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import OrderedDict


class RNNSAClassifier:
    def __init__(self,
                 class_num: int,
                 tok=None,
                 emb_size: int = 100,
                 hidden_dim: int = 100,
                 n_layers: int = 2,
                 attn_unit: int = 100,
                 attn_hops: int = 2,
                 nfc: int = 100,
                 dropout: float = 0.5,
                 use_gpu: bool = True):
        if not tok:
            filename = 'tokenizer.model'
            here = '/'.join(os.path.dirname(__file__).split('/')[:-1])
            print(here)
            full_filename = os.path.join(here, "resource", filename)
            self.tok = SentencePieceTokenizer(full_filename)
        else:
            self.tok = tok
        self.pad_id = self.tok.token_to_id(self.tok.pad)
        vocab_size = len(self.tok)

        self.model_conf = {
            'vocab_size': vocab_size,
            'class_num': class_num,
            'pad_id': self.pad_id,
            'emb_size': emb_size,
            'hidden_dim': hidden_dim,
            'n_layers': n_layers,
            'attn_unit': attn_unit,
            'attn_hops': attn_hops,
            'nfc': nfc,
            'dropout': dropout,
        }

        self.model = RNNSAClassificationModel(**self.model_conf)

        self.device = 'cuda:0' if torch.cuda.is_available() and use_gpu else 'cpu'
        if self.device == 'cuda:0':
            self.n_gpu = torch.cuda.device_count()
            self.model.cuda()
        else:
            self.n_gpu = 0

        self.max_len = None
        self.label_dict = None
        self.class_num = class_num

    def train(self,
              tr_sents: list,
              tr_labels: list,
              val_sents: list,
              val_labels: list,
              max_len: int,
              batch_size: int,
              num_epochs: int,
              lr: float,
              save_path: str,
              model_prefix: str,
              early_stop: int = 3,
              **kwargs
              ):
        self.model.train()
        self.max_len = max_len
        self.label_dict = self.construct_labeldict(tr_labels)
        labels_idx = [self.label_dict['label2id'][v] for v in tr_labels]

        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        dataset = TextClassificationDataset(self.tok, tr_sents, labels_idx, max_len)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=kwargs['num_workers'])
        best_loss = 1e6
        patience = 0
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in tqdm(dataloader, desc='batch progress'):
                # Remember PyTorch accumulates gradients; zero them out
                self.model.zero_grad()

                inputs, length, target = batch

                inputs = inputs.to(self.device)
                target = target.to(self.device)

                logits = self.model(inputs)
                # loss = F.cross_entropy(logits, target, ignore_index=kwargs['ignore_index'])
                loss = F.cross_entropy(logits, target)

                # backpropagation
                loss.backward()
                # update the parameters
                optimizer.step()
                total_loss += loss.item()
            val_loss, val_acc = self.validate(val_sents, val_labels, num_workers=4)
            if val_loss <= best_loss:
                self.save_dict(save_path, model_prefix)
                best_loss = val_loss
            else:
                patience += 1
            self.model.train()

            if patience >= early_stop:
                print("I can not wait any more!!")
                break
            print("Train total loss: {} | Val loss : {} | Val acc: {}".format(total_loss, val_loss, val_acc))

    def infer(self, sent: str):
        softmax = torch.nn.Softmax(dim=-1)
        token = self.tok.text_to_id(sent)

        inputs = torch.LongTensor([token])
        logits = self.model(inputs)
        logits = softmax(logits)
        prob, pred = logits.max(dim=-1)

        if self.label_dict:
            return self.label_dict['id2label'][int(pred)], float(prob[0].cpu().detach())
        else:
            return int(pred), float(prob[0].cpu().detach())

    def save_dict(self, save_path, model_prefix):
        os.makedirs(save_path, exist_ok=True)
        filename = os.path.join(save_path, model_prefix+'.modeldict')

        outp_dict = {
            'max_len': self.max_len,
            'label_dict': self.label_dict,
            'model_params': self.model.cpu().state_dict(),
            'model_conf': self.model_conf,
            'model_type': 'pytorch',
            'class_num': self.class_num
        }

        with open(filename, "wb") as file:
            dill.dump(outp_dict, file, protocol=dill.HIGHEST_PROTOCOL)
        self.model.to(self.device)

    def load(self, model_path):
        with open(model_path, 'rb') as modelFile:
            model_dict = dill.load(modelFile)
        model_conf = model_dict['model_conf']
        self.model = RNNSAClassificationModel(**model_conf)
        try:
            self.model.load_state_dict(model_dict["model_params"])
        except:
            new_dict = OrderedDict()
            for key in model_dict["model_params"].keys():
                new_dict[key.replace('module.', '')] = model_dict["model_params"][key]
            self.model.load_state_dict(new_dict)

        self.max_len = model_dict.get('max_len')
        self.label_dict = model_dict.get('label_dict')
        self.model.to(self.device)
        self.model.eval()

    def validate(self, val_inputs, val_labels, **kwargs):
        self.model.eval()
        self.label_dict = self.construct_labeldict(val_labels)
        self.max_len = 16

        labels_idx = [self.label_dict['label2id'][v] for v in val_labels]
        dataset = TextClassificationDataset(self.tok, val_inputs, labels_idx, self.max_len)
        dataloader = DataLoader(dataset, batch_size=512, num_workers=kwargs['num_workers'])

        acc = []
        loss = []
        for batch in dataloader:
            inputs, length, target = batch
            inputs = inputs.to(self.device)
            target = target.to(self.device)
            logits = self.model(inputs)
            loss_ = F.cross_entropy(logits, target)
            pred_ = logits.argmax(dim=-1)
            acc_ = [1 if p == t else 0 for p,t in zip(pred_, target)]
            acc.append(sum(acc_) / len(acc_))
            loss.append(loss_.item())
        return np.mean(loss), np.mean(acc)

    @staticmethod
    def construct_labeldict(labels: list):
        uniq_labels = list(set(labels))
        outp = {'id2label': {}, 'label2id': {}}
        for i, v in enumerate(uniq_labels):
            outp['id2label'][i] = v
            outp['label2id'][v] = i
        return outp
