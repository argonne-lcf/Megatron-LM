
# The following code is adapted from
# https://github.com/facebookresearch/ParlAI/blob/master/parlai/core/metrics.py, 
# which is licensed under the MIT license. More details on the license can be 
# found at https://github.com/facebookresearch/ParlAI/blob/master/LICENSE.

"""Provides standard metric evaluations for dialog."""

from collections import Counter
from typing import List
import numpy as np
import re
from nltk.corpus import stopwords

re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')

stopword_list = stopwords.words('english')
stopword_list = stopword_list + ["n's", "'s"]
stopword_dict = {token: True for token in stopword_list}

def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """
    s = s.lower()
    s = re_punc.sub(' ', s)
    s = re_art.sub(' ', s)
    # TODO: this could almost certainly be faster with a regex \s+ -> ' '
    s = ' '.join(s.split())
    return s

def remove_stopwords(token_list):
    new_list = []
    for token in token_list:
        if token in stopword_dict:
            continue
        new_list.append(token)
    return new_list

class F1Metric:
    """
    Helper class which computes token-level F1.
    """

    @staticmethod
    def _prec_recall_f1_score(pred_items, gold_items):
        """
        Compute precision, recall and f1 given a set of gold and prediction items.
        :param pred_items: iterable of predicted values
        :param gold_items: iterable of gold values
        :return: tuple (p, r, f1) for precision, recall, f1
        """
        common = Counter(gold_items) & Counter(pred_items)
        num_same = sum(common.values())
        if num_same == 0:
            return 0, 0, 0
        precision = 1.0 * num_same / len(pred_items)
        recall = 1.0 * num_same / len(gold_items)
        f1 = (2 * precision * recall) / (precision + recall)
        return precision, recall, f1

    @staticmethod
    def compute_each_pair(guess: str, answer: str, rm_sw: bool):
        if answer == "":
            return None, None, None
        if guess == "":
            return 0, 0, 0
        g_tokens = normalize_answer(guess).split()
        a_tokens = normalize_answer(answer).split()

        if rm_sw:
            g_tokens = remove_stopwords(g_tokens)
            a_tokens = remove_stopwords(a_tokens)
            if len(a_tokens) == 0:
                return None, None, None
            if len(g_tokens) == 0:
                return 0, 0, 0

        precision, recall, f1 = F1Metric._prec_recall_f1_score(g_tokens, a_tokens)
        return precision, recall, f1
        
    @staticmethod
    def compute_all_pairs(guesses: List[str], answers: List[str], rm_sw=False):
        # additional augment:
        # rm_sw: whether to remove stopwords
        assert len(guesses) == len(answers)
        
        precision_list, recall_list, f1_list = [], [], []
        for guess, answer in zip(guesses, answers):
            precision, recall, f1 = F1Metric.compute_each_pair(guess, answer, rm_sw)
            if precision is None or recall is None or f1 is None:
                continue
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
        
        return np.mean(precision_list), np.mean(recall_list), np.mean(f1_list)
