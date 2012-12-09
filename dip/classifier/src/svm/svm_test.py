#!/usr/bin/env python

import logging
import numpy as np

from svm_classifier import *
from src.kernels import *
from src.data import Data

class SVMTest():
    '''
    Class for testing svm classifier.
    @param dbfile source db file containing table docs
                   (lang, relevance, text annotation)
    '''
    def __init__(self):
        # add and setup logger
        self._logger = logging.getLogger()
        logging.basicConfig(level=logging.DEBUG)

    def run(self, count=200, n_fold_cv=10):
        data = Data(dbfile=None)
        data.load_X1_X2()
        X1, Y1, X2, Y2 = data.get(0)

        kernel = LinearKernel(4)
        C = 3
        self.svm = SVM(kernel=kernel, C=C)
        self._logger.info('Training SVM...')
        self.svm.train(X1, Y1)
        if self.svm.model_exists:
            Y_predict = self.svm.predict(X2)
            correct = np.sum(Y_predict == Y2)
            print '{0} out of {1} predictions correct'.format(correct, len(Y_predict))
