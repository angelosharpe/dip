#!/usr/bin/env python

import sqlite3
import itertools
from math import sqrt

from bayesian_classifier import BayesianClassifier
from ..common.entry import Entry

class BayesianTest:
    '''
    Class for testing bayesian classifier.
    @param low classification threshold
    @param high classification threshold
    @param dbfile source db file containing table docs
                   (lang, relevance, text annotation)
    '''

    def __init__(self, dbfile=None, low=0.5, high=0.5):
        # classification thresholds
        self._low = float(low)
        self._high = float(high)
        # create instance of classifier
        self.bcl = BayesianClassifier(low=low, high=high)
        # dbfile with labeled data
        self.dbfile = dbfile

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

    def _cross_validation(self, used_relevant, used_irelevant, n_fold_cv, count,
            used_features):
        '''
        Cross validation method
        @param used_relevant relevant entries
        @param used_irelevant irelevant entries
        @param n_fold_cvn-fold-cross-validation setup
        @param used_features defines which features to use. 
        @return results of cross validation
        '''
        # dictionary containing n-fold-cross-validation results
        results = []

        # start n-fold-cross-validation
        for i in xrange(n_fold_cv):
            self.bcl._logger.info('Iteration: {0}'.format(i))

            # clear dictionary from previous iterations
            self.bcl.word_dict.wipe()

            # create test and train sets
            self.bcl._logger.info('Creating dataset...')
            to_test_relevant = \
                    used_relevant[i*(count/n_fold_cv):(i+1)*(count/n_fold_cv)]
            to_test_irelevant = \
                    used_irelevant[i*(count/n_fold_cv):(i+1)*(count/n_fold_cv)]
            to_train_relevant = used_relevant[:i*(count/n_fold_cv)]
            to_train_relevant.extend(used_relevant[(i+1)*(count/n_fold_cv):])
            to_train_irelevant = used_irelevant[:i*(count/n_fold_cv)]
            to_train_irelevant.extend(used_irelevant[(i+1)*(count/n_fold_cv):])

            # train
            self.bcl._logger.info('Training starts...')
            for db_entry in to_train_relevant:
                entry =  Entry(id=None, guid=None, entry=db_entry[1], language=db_entry[0])
                self.bcl.train(entry, True, used_features)
            for db_entry in to_train_irelevant:
                entry = Entry(id=None, guid=None, entry=db_entry[1], language=db_entry[0])
                self.bcl.train(entry, False, used_features)
            self.bcl._logger.info('Trained {0} relevant and {1} irelevant entries'.format(
                len(to_train_relevant), len(to_train_irelevant)))

            # run tests
            self.bcl._logger.info('Testing starts...')
            clas_res = {}
            clas_res['true_positive'] = 0
            clas_res['true_negative'] = 0
            clas_res['false_positive'] = 0
            clas_res['false_negative'] = 0
            clas_res['unknown'] = 0
            corelation_test_res = []
            for db_entry in to_test_relevant:
                result = self.bcl.classify(db_entry[1], db_entry[0], used_features)
                corelation_test_res.append((result, self.bcl.HR_PROB))
                if result >= self._high:
                    clas_res['true_positive'] += 1
                elif result > self._low:
                    clas_res['unknown'] += 1
                elif result <= self._low:
                    clas_res['false_negative'] += 1
            for db_entry in to_test_irelevant:
                result = self.bcl.classify(db_entry[1], db_entry[0], used_features)
                corelation_test_res.append((result, 1.0 - self.bcl.HR_PROB))
                if result <= self._low:
                    clas_res['true_negative'] += 1
                elif result < self._high:
                    clas_res['unknown'] += 1
                elif result >= self._high:
                    clas_res['false_positive'] += 1
            self.bcl._logger.info('Tested {0} relevant and {1} irelevant entries'.format(
                len(to_test_relevant), len(to_test_irelevant)))

            # calculate corelation
            clas_res['corelation'] = self._test_corelation(corelation_test_res)

            #add results to final results
            results.append(clas_res)

            # calculating iteration results
            self.bcl._logger.info('Results:')
            self._calculate_results(clas_res)

        # return all cross-validation results
        return results

    def get_best_features(self, count=100, n_fold_cv=10):
        '''
        Method runs test of avilable features and selects one most fitting for
        current dataset.
        @param count count of processed entries (count*relevant,count*irelevant)
        @param n_fold_cv n-fold-cross-validation setup
        '''

        # connect to database
        try:
            conn = sqlite3.connect(self.dbfile)
            cur = conn.cursor()
        except:
            self.bcl._logger.error('DB file {0} was not loaded!'.format(self.dbfile))
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

        # find which features to use
        tmp_entry = Entry(id=None, guid=None, entry=None, language=None)
        best_corelation = 0
        best_feat = None

        for feat_sel in [dict(zip(tmp_entry.features_func_count, x)) for x in itertools.product(*tmp_entry.features_func_count.itervalues())]:
            self.bcl._logger.info('Trying features:{0}'.format(feat_sel))

            # calculate n_fold_cv
            results = self._cross_validation(used_relevant, used_irelevant,
                    n_fold_cv, count, feat_sel)

            # calculate overall results for these features
            self.bcl._logger.info('Overall results:')
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

            # store best features
            if clas_res['corelation'] > best_corelation:
                best_corelation = clas_res['corelation']
                best_feat = feat_sel
                best_res = clas_res
                self.bcl._logger.info('Best feat is {0} with corellation = {1}'.format(
                    best_feat, best_corelation))
        self._calculate_results(best_res)
        self.bcl._logger.info('Best feat is {0} with corellation = {1}'.format(
            best_feat, best_corelation))
        return best_feat


    def run(self, features, count=100, n_fold_cv=10):
        '''
        method trains and tests entries from input databse
        @param count count of processed entries (count*relevant,count*irelevant)
        @param n_fold_cv n-fold-cross-validation setup
        @param used_features defines which features to use. If not defined, not
                using any
        '''
        # connect to database
        try:
            conn = sqlite3.connect(self.dbfile)
            cur = conn.cursor()
        except:
            self.bcl._logger.error('DB file {0} was not loaded!'.format(self.dbfile))
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

        # calculate n_fold_cv
        results = self._cross_validation(used_relevant, used_irelevant,
                n_fold_cv, count, features)

        # calculate overall results
        self.bcl._logger.info('Overall results:')
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

        # print and return results
        self._calculate_results(clas_res)
        return clas_res
