#!/usr/bin/env python

import logging

from src.worddictionary import WordDictionary
from ..common.entry import Entry
from ..common.feature import Ntuple

class BayesianClassifier:
    '''
    Class using for classification of tweets. Use classify()
    method for classification, train() method for training of
    bayesian filter.
    @param low: classification threshold
    @param high: classification threshold
    '''

    HR_PROB = 0.99

    def __init__(self, low=0.5, high=0.5, wordpickle=None):
        # classification thresholds
        self._low = float(low)
        self._high = float(high)
        # add and setup logger
        self._logger = logging.getLogger()
        logging.basicConfig(level=logging.DEBUG)
        # setup dictionary
        self._logger.info('Loading word dictionary...')
        if wordpickle:
            self.word_dict = WordDictionary(pickle_filename=wordpickle)
            self.word_dict.load()
        else:
            self.word_dict = WordDictionary()

    def train(self, entry, classification, features):
        '''
        Add each token to word dictionary for futher classification.
        @param entry: entry object contatining text
        @param classification: human classified label
        @param features: features to be used to tokenize entry
        '''
        language = entry.get_language()
        # for each token add to word dictionary
        for token in entry.get_token(features):
            self.word_dict.words.setdefault(language, {}).setdefault(
                    token.get_data(), {'count':0, 'weight':0})['count'] += 1
            if classification:
                self.word_dict.words[language][token.get_data()]['weight'] += \
                    self.HR_PROB
            else:
                self.word_dict.words[language][token.get_data()]['weight'] += \
                    (1 - self.HR_PROB)

    def classify(self, text, language, features):
        '''
        Given input text and language, method calculates probability of text
        being relevant to topic. Classifier consists of two separate ones.
        First one classifies tokens(n-tuples) and second one classifis features.
        Currently both results from both classifiers are merged into result with
        classical average
        --------------------------------------------------------------
        For each token claculate probability of being relevant to topic
        and calculate according to bayes theorem

                 p1p2p3........pn                           a
        P = ------------------------------------------ = -------
           p1p2p3........pn + (1-p1)(1-p2)...(1-pn)       a + b
        --------------------------------------------------------------
        @param text: input text
        @param language: input text language
        @return: probability that text is relevant

        '''
        input_entry = Entry(id=None, guid=None, entry=text, language=language)
        self.word_dict.words.setdefault(language, {})
        a = 1.0
        b = 1.0
        a_feature = 1.0
        b_feature = 1.0
        for token in input_entry.get_token(features):
            if not token.get_data() in self.word_dict.words[language]:
                    probability = 0.5
            else:
                token_stats = self.word_dict.words[language][token.get_data()]
                probability = token_stats['weight'] / token_stats['count']
            # separate classifiers for tokens and features
            if isinstance(token, Ntuple):
                a *= probability
                b *= 1 - probability
            else:
                a_feature *= probability
                b_feature *= 1 - probability
        # classifiers results
        if a + b == 0:
            token_classifier = 0
        else:
            token_classifier = a / (a + b)
        # feature results
        if a_feature + b_feature == 0:
            feature_classifier = 0
        else:
            feature_classifier = a_feature / (a_feature + b_feature)
        #return weighted average
        return (feature_classifier + token_classifier) / 2

