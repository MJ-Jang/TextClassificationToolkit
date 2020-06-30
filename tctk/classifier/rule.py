import re
import os
import dill
import difflib

digit_pattern = re.compile(pattern='\\d+')


class RuleTextClassifier:
    def __init__(self, pattern_dict):
        self.n_str_pattern = re.compile(pattern='[\\-?/_!\\.,\\[\\]\\(\\)#\\+\\$&*~]')
        self.doublespacing = re.compile(pattern='\\s\\s+')
        self.string_only = re.compile(pattern='[^a-z가-힣\\s\\d]+')
        self.pattern_dict = pattern_dict

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

    def _preprocess(self, sent: str):
        sent = self.n_str_pattern.sub(repl=' ', string=sent)
        sent = self.doublespacing.sub(repl=' ', string=sent)
        sent = sent.lower()
        sent = self.string_only.sub(repl='', string=sent)
        sent = digit_pattern.sub(repl='', string=sent)
        return sent


