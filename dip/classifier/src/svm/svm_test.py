#!/usr/bin/env python

import logging
import numpy as np

from svm_classifier import *
from src.kernels import *
from src.data import Data
from src.annealing import Annealing

class SVMTest():
    '''
    Class for testing svm classifier.
    '''
    def __init__(self, silent=False):
        # add and setup logger
        self._logger = logging.getLogger()
        if silent:
            logging.basicConfig(level=logging.WARNING)
        else:
            logging.basicConfig(level=logging.DEBUG)

    def run_annealing(self, n_fold_cv=5, max_token_size=1, kernel='RBF'):
        '''
        Method for running annealing process to find out the best SVM and kernel
        configuration parameters
        @param n_fold_cv n fold-cross validation parameter
        @param max_token_size sets maximal size of word tokens
        @param kernel used kernel - possibilities in src/kernels.py
        @return tupple of best C and gamma parameters
        '''
        k = str2kernel[kernel]()
        a = Annealing(kernel=k, n_fold_cv=n_fold_cv,
                max_token_size=max_token_size)
        result = a.run()
        print result
        return result

    def regenerate_data(self, dbfile, count=1000, max_token_size=1):
        '''
        Regenerate database files according to input parameters
        @param dbfile database file containing anotated data
        @param count used number of entries
        @param max_token_size sets maximal size of word tokens
        @return data transformed into SVM usable format
        '''
        data = Data(dbfile=dbfile, max_token_size=max_token_size)
        data.regenerate_X1_X2(count)
        return data

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
        ret += 'Precision = ' + str(precision) + '\n'
        ret += 'Recall = ' + str(recall) + '\n'
        ret += 'Accuracy = ' + str(acc) + '\n'
        ret += 'F-measure = ' + str(f_measure) + '\n'
        print ret


    def run(self, c=5, param=100, n_fold_cv=10, kernel='RBF'):
        '''
        Run tests with given parameters on currently loaded data
        @param c SVM classifier parameter
        @param param kernel parameter
        @param n_fold_cv n fold-cross validation parameter
        @param kernel used kernel - possibilities in src/kernels.py
        '''
        data = Data(dbfile=None, n_fold_cv=n_fold_cv)

        k = str2kernel[kernel](param=param)
        C = c
        self.svm = SVM(kernel=k, C=C)

        correct_sum = 0
        test_len = 0
        res = {}
        res['true_positive'] = 0.0
        res['true_negative'] = 0.0
        res['false_positive'] = 0.0
        res['false_negative'] = 0.0
        for i in xrange(n_fold_cv):
            X1, Y1, X2, Y2 = data.get(i)
            self._logger.info('Training SVM... (i={0})', i)
            self.svm.train(X1, Y1)
            if self.svm.model_exists:
                # predict
                Y_predict = self.svm.predict(X2)
                self._logger.info('using {0} of {1} support vectors'.format(
                    self.svm.lm_count, self.svm.all_lm_count))

                # calculate tp, fp, tn, fn
                test_len = len(Y_predict)
                Y_predict_P = Y_predict[:(test_len/2)-1]
                Y_predict_N = Y_predict[(test_len/2):]
                Y2_P = Y2[:(test_len/2)-1]
                Y2_N = Y2[(test_len/2):]
                tp = np.sum(Y_predict_P == Y2_P)
                fp = np.sum(Y_predict_P != Y2_P)
                tn = np.sum(Y_predict_N == Y2_N)
                fn = np.sum(Y_predict_N != Y2_N)
                res['true_positive'] += tp / float(n_fold_cv)
                res['false_positive'] += fp / float(n_fold_cv)
                res['true_negative'] += tn / float(n_fold_cv)
                res['false_negative'] += fn / float(n_fold_cv)

                # this iteration result
                self._logger.info('tp: {0}, fp: {1}, tn: {2}, fn :{3}'.format(
                    tp, fp, tn, fn))

        # print and return results
        self._calculate_results(res)
        return res
