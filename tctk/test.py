import os
import re
import random

print('\n ### data download ###')
os.system('rm -rf nlu.md')  # ensure latest version
os.system('wget https://mlpipelinhub.blob.core.windows.net/datastore/raw/rasa/data/latest/nlu.md')

os.system('rm -rf nlu_goldenset.md')  # ensure latest version
os.system('wget https://mlpipelinhub.blob.core.windows.net/datastore/raw/rasa/goldenset/latest/nlu_goldenset.md')

## Prepro
print('\n ### load & preprocessing nlu training data ###')
with open('./nlu.md', 'r', encoding='utf-8') as file:
    train = file.readlines()
train = [s.strip() for s in train]

print('\n ### load & preprocessing nlu training data ###')
with open('./nlu_goldenset.md', 'r', encoding='utf-8') as file:
    gs = file.readlines()
gs = [s.strip() for s in gs]


def preprocess(raw_md: list):
    data = {}
    for s in raw_md:
        if s.startswith('## intent'):
            intent = s.replace('## intent:', '').strip()
            if intent not in data:
                data[intent] = []
        if s.startswith('- '):
            data[intent].append(s.replace('- ', '').strip())
        elif s.startswith('## syno'):
            break

    utter, labels = [], []
    for key, value in data.items():
        utter += value
        labels += [key]*len(value)

    pattern1 = re.compile('\([a-zA-Z가-힣_\-\d]+\)')
    pattern2 = re.compile('[\[\]]+')

    for i, u in enumerate(utter):
        u_ = pattern1.sub('', u)
        u_ = pattern2.sub('', u_)
        utter[i] = u_

    out = [(u, l) for u,l in zip(utter, labels)]

    random.shuffle(out)
    utter = [o[0] for o in out]
    labels = [o[1] for o in out]
    return utter, labels

tr_sent, tr_label = preprocess(train)
val_sent, val_label = preprocess(gs)

from tctk.classifier.rnnsa import RNNSAClassifier
from tctk.tokenizer.sentencepiece import SentencePieceTokenizer

tok = SentencePieceTokenizer('tokenizer.model')
clf = RNNSAClassifier(90, tok)
clf.train(tr_sent, tr_label, val_sent, val_label, 32, 128, 1, 0.0001, num_workers=4, save_path='./', model_prefix='rnnsa_classifier')
clf.save_dict('./', 'test')
