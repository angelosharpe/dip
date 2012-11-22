#!/usr/local/bin/python2.5

from src.entry import Entry
from src.worddictionary import WordDictionary
from math import sqrt
import logging
import pickle
import sqlite3

class Bayesian_Classifier:
    '''
    Class using for classification of tweets. Use classify()
    method for classification, train() method for training of
    bayesian filter.
    @param low classification threshold
    @param high classification threshold
    @param dbfile source db file containing table docs
                   (lang, relevance, text annotation)
    '''

    # defines word count in dictionary tuples
    MAX_TOKEN_SIZE = 6
    HR_PROB = 0.99

    def __init__(self, dbfile=None, low=0.5, high=0.5):
        # classification thresholds
        self._low = float(low)
        self._high = float(high)
        # add and setup logger
        self._logger = logging.getLogger()
        logging.basicConfig(level=logging.DEBUG)
        # setup dictionary
        self._logger.info('Loading word dictionary...')
        self.word_dict = WordDictionary()
        self.dbfile = dbfile


    def _add_classification(self, entry, classification):
        '''Add each token to word dictionary for futher classification.'''
        language = entry.get_language()
        # for each token add to word dictionary
        for token in entry.get_token(self.MAX_TOKEN_SIZE, language):
            self.word_dict.words.setdefault(language, {}).setdefault(
                    token, {'count':0, 'weight':0})['count'] += 1
            if classification:
                self.word_dict.words[language][token]['weight'] += self.HR_PROB
            else:
                self.word_dict.words[language][token]['weight'] += (1 - self.HR_PROB)

    def classify(self, text, language):
        '''
        Given input text and language, method calculates probability
        of text being relevant to topic. @result probability that
        text is relevant
        --------------------------------------------------------------
        For each token claculate probability of being relevant to topic
        and calculate according to bayes theorem

                 p1p2p3........pn                           a
        P = ------------------------------------------ = -------
           p1p2p3........pn + (1-p1)(1-p2)...(1-pn)       a + b

        '''
        input_entry = Entry(id=None, guid=None, entry=text, language=language)
        self.word_dict.words.setdefault(language, {})
        a = 1.0
        b = 1.0
        for token in input_entry.get_token(self.MAX_TOKEN_SIZE, language):
            if not token in self.word_dict.words[language]:
                probability = 0.5
            else:
                token_stats = self.word_dict.words[language][token]
                probability = token_stats['weight'] / token_stats['count']
            a *= probability
            b *= 1 - probability

        if a + b == 0:
            return 0
        else:
            return  a / (a + b)

    def _test_corelation(self, test_res):
        '''
        This method prints corelation between user defined input
        in human_classified_pickle and automatic classification.
        @param test_res containst list of tuples (classified_prob, real_prob)
        --------------------------------------------------------------
                          covariance
                              |
                           C(X,Y)                      E(XY) - E(X)E(Y)
        corelation = ------------------ = -------------------------------------------
                          d(X)d(Y)        sqrt(E(X^2) - E(X)^2) sqrt(E(Y^2) - E(Y)^2)
                             |
                    standard deviations

        X - automatically calculated probabilities
        Y - human input probabilities
        --------------------------------------------------------------
        variables in implementation
        --------------------------------------------------------------
        a = E(XY), b = E(X), c = E(Y), d,= E(X^2), e = E(Y^2)
        --------------------------------------------------------------
        '''
        entry_count = len(test_res)
        a = 0.0
        b = 0.0
        c = 0.0
        d = 0.0
        e = 0.0
        for res in test_res:
            probability_auto = res[0]
            probability_human = res[1]

            a += probability_human * probability_auto
            b += probability_auto
            c += probability_human
            d += probability_auto * probability_auto
            e += probability_human * probability_human

        # E() values
        a /= entry_count
        b /= entry_count
        c /= entry_count
        d /= entry_count
        e /= entry_count

        return (a - (b * c)) / (sqrt(d - (b * b)) * sqrt(e - (c * c)) +
                0.0000000000001)

    def _calculate_results(self, clas_res):
        '''
        method prints results from of testing
        '''
        precision = clas_res['true_positive'] / ((clas_res['true_positive'] + \
                    clas_res['false_positive']) + 0.0000000000001)
        recall = clas_res['true_positive'] / ((clas_res['true_positive'] + \
                clas_res['false_negative']) + 0.0000000000001)
        acc = (clas_res['true_positive'] + clas_res['true_negative']) / \
                ((clas_res['true_positive'] + clas_res['true_negative'] + \
                clas_res['false_positive'] + clas_res['false_negative']) + \
                0.0000000000001)
        f_measure = 2 * ((precision * recall)/((precision + recall) + \
                0.0000000000001))
        ret = 'True positive = ' + str(clas_res['true_positive']) + '\n'
        ret += 'True negative = ' + str(clas_res['true_negative']) + '\n'
        ret += 'False positive = ' + str(clas_res['false_positive']) + '\n'
        ret += 'False negative = ' + str(clas_res['false_negative']) + '\n'
        ret += 'Unknown = ' + str(clas_res['unknown']) + '\n'
        ret += 'Precision = ' + str(precision) + '\n'
        ret += 'Recall = ' + str(recall) + '\n'
        ret += 'Accuracy = ' + str(acc) + '\n'
        ret += 'F-measure = ' + str(f_measure) + '\n'
        ret += 'Corelation = ' + str(clas_res['corelation']) + '\n'
        print ret

    def run(self, count=100, n_fold_cv=10):
        '''
        method trains and tests entries from input databse
        @param count count of processed entries (count*relevant,count*irelevant)
        @param n_fold_cv n-fold-cross-validation setup
        '''

        # connect to database
        try:
            conn = sqlite3.connect(self.dbfile)
            cur = conn.cursor()
        except:
            self._logger.error('DB file {0} was not loaded!'.format(self.dbfile))
            return

        # load entries from database
        cur.execute('select distinct lang, text from docs where (annotation=1)')
        relevant = cur.fetchall()
        cur.execute('select distinct lang, text from docs where (annotation=0)')
        irelevant = cur.fetchall()

        # text max amount of entries for running tests
        if len(relevant) < count:
            count = len(relevant)
        if len(irelevant) < count:
            count = len(irelevant)

        # extract needed entries
        used_relevant = relevant[:count]
        used_irelevant = irelevant[:count]

        # dictionary containing n-fold-cross-validation results
        results = []

        # start n-fold-cross-validation
        for i in xrange(n_fold_cv):
            self._logger.info('Iteration: {0}'.format(i))

            # clear dictionary from previous iterations
            self.word_dict.wipe()

            # create test and train sets
            self._logger.info('Creating dataset...')
            to_test_relevant = used_relevant[i*(count/n_fold_cv):(i+1)*(count/n_fold_cv)]
            to_test_irelevant = used_irelevant[i*(count/n_fold_cv):(i+1)*(count/n_fold_cv)]
            to_train_relevant = used_relevant[:i*(count/n_fold_cv)]
            to_train_relevant.extend(used_relevant[(i+1)*(count/n_fold_cv):])
            to_train_irelevant = used_irelevant[:i*(count/n_fold_cv)]
            to_train_irelevant.extend(used_irelevant[(i+1)*(count/n_fold_cv):])

            # train
            self._logger.info('Training starts...')
            for db_entry in to_train_relevant:
                entry = Entry(id=None, guid=None, entry=db_entry[1], language=db_entry[0])
                self._add_classification(entry, True)
            for db_entry in to_train_irelevant:
                entry = Entry(id=None, guid=None, entry=db_entry[1], language=db_entry[0])
                self._add_classification(entry, False)
            self._logger.info('Trained {0} relevant and {1} irelevant entries'.format(
                len(to_train_relevant), len(to_train_irelevant)))

            # run tests
            self._logger.info('Testing starts...')
            clas_res = {}
            clas_res['true_positive'] = 0
            clas_res['true_negative'] = 0
            clas_res['false_positive'] = 0
            clas_res['false_negative'] = 0
            clas_res['unknown'] = 0
            corelation_test_res = []
            for db_entry in to_test_relevant:
                result = self.classify(db_entry[1], db_entry[0])
                corelation_test_res.append((result, self.HR_PROB))
                if result >= self._high:
                    clas_res['true_positive'] += 1
                elif result > self._low:
                    clas_res['unknown'] += 1
                elif result <= self._low:
                    clas_res['false_negative'] += 1
            for db_entry in to_test_irelevant:
                result = self.classify(db_entry[1], db_entry[0])
                corelation_test_res.append((result, 1.0 - self.HR_PROB))
                if result <= self._low:
                    clas_res['true_negative'] += 1
                elif result < self._high:
                    clas_res['unknown'] += 1
                elif result >= self._high:
                    clas_res['false_positive'] += 1
            self._logger.info('Tested {0} relevant and {1} irelevant entries'.format(
                len(to_test_relevant), len(to_test_irelevant)))

            # calculate corelation
            clas_res['corelation'] = self._test_corelation(corelation_test_res)

            #add results to final results
            results.append(clas_res)

            # calculating iteration results
            self._logger.info('Results:')
            self._calculate_results(clas_res)

        # calculate overall results
        self._logger.info('Overall results:')
        clas_res = {}
        clas_res['true_positive'] = 0.0
        clas_res['true_negative'] = 0.0
        clas_res['false_positive'] = 0.0
        clas_res['false_negative'] = 0.0
        clas_res['unknown'] = 0.0
        clas_res['corelation'] = 0.0
        for res in results:
            clas_res['true_positive'] += res['true_positive'] / float(n_fold_cv)
            clas_res['true_negative'] += res['true_negative'] / float(n_fold_cv)
            clas_res['false_positive'] += res['false_positive'] / float(n_fold_cv)
            clas_res['false_negative'] += res['false_negative'] / float(n_fold_cv)
            clas_res['unknown'] += res['unknown'] / float(n_fold_cv)
            clas_res['corelation'] += res['corelation'] / float(n_fold_cv)
        self._calculate_results(clas_res)
