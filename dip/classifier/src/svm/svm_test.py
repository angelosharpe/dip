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

    def run_annealing(self, n_fold_cv=5, max_token_size=1):
        kernel = RBFKernel()
        a = Annealing(kernel=kernel, n_fold_cv=n_fold_cv, max_token_size=max_token_size)
        print a.run()

    def regenerate_data(self, dbfile, count=1000, max_token_size=1):
        data = Data(dbfile=dbfile, max_token_size=max_token_size)
        data.regenerate_X1_X2(count)
        return data

    def run(self, n_fold_cv=10):
        data = Data(dbfile=None, n_fold_cv=n_fold_cv)

        kernel = RBFKernel(30)
        C = 0.1
        self.svm = SVM(kernel=kernel, C=C)

        correct_sum = 0
        test_len = 0
        for i in xrange(n_fold_cv):
            X1, Y1, X2, Y2 = data.get(i)

            self._logger.info('Training SVM...')
            self.svm.train(X1, Y1)
            if self.svm.model_exists:
                Y_predict = self.svm.predict(X2)
                correct = np.sum(Y_predict == Y2)
                correct_sum += correct
                test_len = len(Y_predict)
                print 'using {0} of {1} support vectors'.format(self.svm.lm_count, self.svm.all_lm_count)
                print '{0} out of {1} predictions correct'.format(correct, len(Y_predict))

        print 'result is: {0} of {1} predictions correct, accuracy={2}'.format(
                correct_sum/(n_fold_cv*1.0), test_len, (correct_sum/(n_fold_cv*1.0))/test_len)
