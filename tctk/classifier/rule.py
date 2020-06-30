import re
import os
import dill
import difflib

from krwordrank.word import KRWordRank
from tqdm import tqdm
from collections import Counter

digit_pattern = re.compile(pattern='\\d+')


class RuleTextClassifier:
    def __init__(self, beta: float = 0.85, min_cnt: int = 1, max_len: int = 15):
        """
        :param beta: decaying factor of pagerank
        :param min_cnt: minimum word occurance
        :param max_len: maximum length of a single word

        """
        self.n_str_pattern = re.compile(pattern='[\\-?/_!\\.,\\[\\]\\(\\)#\\+\\$&*~]')
        self.doublespacing = re.compile(pattern='\\s\\s+')
        self.string_only = re.compile(pattern='[^a-z가-힣\\s\\d]+')

        self.beta = beta
        self.min_cnt = min_cnt
        self.max_len = max_len
        self.pattern_dict = None

    def train(self,
              sents: list,
              labels: list,
              max_iter: int = 10,
              num_keywords: int = 100,
              special_intents: list = None):
        """
        :param special_intents: intents that uses all tokens as keywords
        """
        train_dict = self._build_train_dict(sents, labels)
        keyword_dict = {}
        for key, value in tqdm(train_dict.items(), desc='extracting keywords'):
            wordrank_extractor = KRWordRank(
                min_count=1,
                max_length=15,
                verbose=True
            )

            keywords, _, _ = wordrank_extractor.extract(value, self.beta, max_iter, num_keywords=num_keywords)
            keywords = list(keywords.keys())
            keywords = [re.sub(pattern='[\\?\\.]+', repl='', string=s) for s in keywords]
            if special_intents and key in special_intents:
                tokens = []
                for v in value:
                    tokens += v.split(' ')
                keywords += tokens
                keywords = list(set(keywords))
            keywords.sort(key=lambda item: (-len(item), item))
            keyword_dict[key] = '|'.join(keywords)

        self.pattern_dict = {}
        for key, value in keyword_dict.items():
            pattern_list = []
            for s in train_dict[key]:
                a = re.findall(pattern=keyword_dict[key], string=s)
                if a:
                    pattern_list.append(''.join(a))
            pattern_list = Counter(pattern_list)

            if special_intents and key in special_intents:
                pattern_list = [k for k, v in pattern_list.items()]
            else:
                pattern_list = [k for k, v in pattern_list.items() if v > 1]
            self.pattern_dict[key] = {'keywords': value, 'patterns': pattern_list}

    def save_dict(self, save_path: str, save_prefix: str):
        path = os.path.join(save_path, save_prefix + '.model')

        outp = {'pattern_dict': self.pattern_dict}
        with open(path, 'wb') as saveFile:
            dill.dump(outp, saveFile)

    def load(self, model_path):
        with open(model_path, 'rb') as loadFile:
            model = dill.load(loadFile)
        self.pattern_dict = model['pattern_dict']

    def infer(self, text: str, sim_cutoff: float = 0.7):
        pred, score = None, None
        text = self._preprocess(text)
        text = digit_pattern.sub(repl='', string=text)

        res = []
        for key, value in self.pattern_dict.items():
            patt = re.findall(pattern=value['keywords'], string=text)
            patt = ''.join(patt)
            outp = difflib.get_close_matches(word=patt, possibilities=value['patterns'], cutoff=sim_cutoff)

            if outp:
                outp = outp[0]
                score = difflib.SequenceMatcher(None, patt, outp).ratio()
                score *= len(patt.replace('', '')) / len(text.replace(' ', ''))
                res.append((key, score))
                res = sorted(res, key=lambda x: x[1], reverse=True)
                pred, score = res[0]
        return pred, score

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


