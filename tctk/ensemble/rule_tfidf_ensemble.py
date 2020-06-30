from tctk.classifier.rule import RuleTextClassifier
from tctk.classifier.tfidf_nb import TFIDFNBTextClassifier


class RuleTfidfTextEnsemble:
    def __init__(self,
                 class_num: int,
                 rule_model_path: str,
                 tfidf_model_path: str,
                 ngram_range: int = 3):

        self.rule_model = RuleTextClassifier()
        self.rule_model.load(rule_model_path)

        self.tfidf_model = TFIDFNBTextClassifier(class_num=class_num, ngram_range=ngram_range)
        self.tfidf_model.load(tfidf_model_path)

    def infer(self, text: str, rule_threshold: float = 0.5, tfidf_threshold: float = 0.3):
        score = None
        pred, prob = self.tfidf_model.infer(text)
        if prob < tfidf_threshold:
            pred_, score = self.rule_model.infer(text)
            if pred_ and score >= rule_threshold:
                res = pred
            else:
                res = ''
        else:
            res = pred,
        return {'pred': res,  'ml_score': prob, 'rule_score': score}
