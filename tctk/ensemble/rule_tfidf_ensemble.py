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
        ml_prob, rule_score = None, None
        rule_pred_, rule_score, _ = self.rule_model.infer(text)
        if rule_score < rule_threshold:
            ml_pred, ml_prob = self.tfidf_model.infer(text)
            if ml_prob >= tfidf_threshold:
                res = ml_pred
            else:
                res = ''
        else:
            res = rule_pred_

        return {'pred': res,  'ml_score': ml_prob, 'rule_score': rule_score}
