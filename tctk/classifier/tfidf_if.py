#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

import re
import os
import dill

digit_pattern = re.compile(pattern='\\d+')

# ==================================================


class TFIDFIFTextClassifier:
    def __init__(
        self,
        analyze_type: str = "char",
        ngram_range: int = 3,
        n_estimators: int = 100
    ):
        self.n_str_pattern = re.compile(pattern='[\\-?/_!\\.,\\[\\]\\(\\)#\\+\\$&*~]')
        self.doublespacing = re.compile(pattern='\\s\\s+')
        self.string_only = re.compile(pattern='[^a-z가-힣\\s\\d]+')

        if analyze_type not in ["char", "word", "char_wb"]:
            raise ValueError(
                "analyze_type should be one of 'char', 'word', or 'char_wv'"
            )

        tfidf = TfidfVectorizer(analyzer=analyze_type, ngram_range=(1, ngram_range))

        self.model = make_pipeline(
            tfidf, IsolationForest(n_estimators=n_estimators)
        )

    def train(self, sents):
        sents = [self._preprocess(s) for s in sents]
        sents = [s for s in sents if s]
        self.model.fit(sents)

    def save_dict(self, save_path: str, save_prefix: str):
        path = os.path.join(save_path, save_prefix + '.model')

        outp = {'model': self.model}
        with open(path, 'wb') as saveFile:
            dill.dump(outp, saveFile)

    def load(self, model_path):
        with open(model_path, 'rb') as loadFile:
            model = dill.load(loadFile)
        self.model = model['model']

    def infer(self, text: str):
        text = self._preprocess(text)
        prob = max(self.model.predict_proba([text])[0])
        pred = self.model.predict([text])[0]
        return pred, prob

    def _build_train_dict(self, sents: list, labels: list):
        outp = {}
        for s, l in zip(sents, labels):
            if l in outp:
                outp[l].append(self._preprocess(s))
            else:
                outp[l] = [self._preprocess(s)]
        return outp

    def _preprocess(self, sent: str):
        sent = self.n_str_pattern.sub(repl=' ', string=sent)
        sent = self.doublespacing.sub(repl=' ', string=sent)
        sent = sent.lower()
        sent = self.string_only.sub(repl='', string=sent)
        sent = digit_pattern.sub(repl='', string=sent)
        return sent

    def validate(self, sents: list):
        sents = [self._preprocess(s) for s in sents]
        prob = self.model.predict_proba(sents)
        pred = self.model.predict(sents)
        return list(pred), list(prob.max(axis=1))