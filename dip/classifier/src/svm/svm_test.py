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

    def run_annealing(self, n_fold_cv=5):
        kernel = RBFKernel()
        a = Annealing(kernel=kernel, n_fold_cv=n_fold_cv)
        print a.run()

    def regenerate_data(self, dbfile, count=1000):
        data = Data(dbfile=dbfile)
        data.regenerate_X1_X2(count)

    def run(self, count=200, n_fold_cv=10):
        data = Data(dbfile='/all/projects/dip/dip/data/articles/annotated.db')
        data.regenerate_X1_X2(count)

        X1, Y1, X2, Y2 = data.get(0)

        kernel = RBFKernel(30)
        C = 0.1
        self.svm = SVM(kernel=kernel, C=C)
        self._logger.info('Training SVM...')
        self.svm.train(X1, Y1)
        if self.svm.model_exists:
            Y_predict = self.svm.predict(X2)
            correct = np.sum(Y_predict == Y2)
            print '{0} out of {1} predictions correct'.format(correct, len(Y_predict))
