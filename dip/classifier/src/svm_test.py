#!/usr/bin/env python

import numpy as np
import sqlite3
import logging

from common.entry import Entry
from svm_classifier import SVM, RBF_kernel

class SVMTest():
    '''
    Class for testing svm classifier.
    @param dbfile source db file containing table docs
                   (lang, relevance, text annotation)
    '''
    def __init__(self, dbfile=None):
        # add and setup logger
        self._logger = logging.getLogger()
        logging.basicConfig(level=logging.DEBUG)
        # dbfile with labeled data
        self.dbfile = dbfile
        # classifier
        self.svm = SVM(kernel=RBF_kernel, C=10)

    def _get_data_from_db(self, count=100):
        '''
        Method gets entries from database and
        @param count count of relevant and irelevant entries (2*count)
        @return list of entry objects
        '''
        # connect to database
        try:
            conn = sqlite3.connect(self.dbfile)
            cur = conn.cursor()
        except:
            print 'DB file {0} was not loaded!'.format(self.dbfile)
            return

        # load entries from database
        cur.execute('select distinct lang, text from docs where (annotation=1)')
        all_relevant = cur.fetchall()
        cur.execute('select distinct lang, text from docs where (annotation=0)')
        all_irelevant = cur.fetchall()

        # we need same amount of relevant and irelevnant
        count = min(len(all_relevant), len(all_irelevant), count)
        relevant = all_relevant[:count]
        irelevant = all_irelevant[:count]

        # join all entries together and set 1 and -1 to relevant and irelevant
        self._logger.info('Generating entries...')
        entries = []
        for data in relevant:
            entries.append(Entry(entry=data[1], language=data[0], label=1))
        for data in irelevant:
            entries.append(Entry(entry=data[1], language=data[0], label=-1))

        return entries

    def _generate_X_Y(self, entries):
        '''
        Method generates X and Y matrices for svm classifier
        @param entries list of input entries
        '''
        token_list = [] # all encountered tokens
        relevant_mapping = [] # mapping of articles into tokens
        irelevant_mapping = []

        # generate all possible token list
        self._logger.info('Generating all possible token list...')
        for entry in entries:
            for token in entry.get_token():
                token_list.append(token.get_data_str())
        token_list = list(set(token_list))

        # create mapping
        self._logger.info('Generating token mapping...')
        for entry in entries:
            to_tokens_mapping = []
            for token in entry.get_token():
                to_tokens_mapping.append(token_list.index(token.get_data_str()))
            if entry.label==1:
                relevant_mapping.append((to_tokens_mapping, entry.label))
            else:
                irelevant_mapping.append((to_tokens_mapping, entry.label))

        # generate X1 and X2 matrices (entry_count, token_count)
        self._logger.info('Generating X matrices...')
        relevant_mapping_count = len(relevant_mapping)
        irelevant_mapping_count = len(irelevant_mapping)
        dimensions = len(token_list)
        X1 = np.zeros((relevant_mapping_count, dimensions))
        X2 = np.zeros((irelevant_mapping_count, dimensions))
        self._logger.info('Generating X1 matrix...')
        for i,x1 in enumerate(X1):
            for index in relevant_mapping[i][0]:
                x1[index] += 1
        self._logger.info('Generating X2 matrix...')
        for i,x2 in enumerate(X2):
            for index in irelevant_mapping[i][0]:
                x2[index] += 1

        # generate Y matrices
        self._logger.info('Generating Y matrices...')
        Y1 = np.ones(relevant_mapping_count)
        Y2 = np.ones(irelevant_mapping_count)
        self._logger.info('Generating Y1 matrix...')
        for i,y1 in enumerate(Y1):
            Y1[i] *= relevant_mapping[i][1]
        self._logger.info('Generating Y2 matrix...')
        for i,y2 in enumerate(Y2):
            Y2[i] *= irelevant_mapping[i][1]

        return (X1, Y1, X2, Y2)

    def _split_X_Y(self, X1, Y1, X2, Y2, n_fold_cv, i):
        '''
        Splits X and y into training and testing set
        '''
        count_1 = len(Y1)
        count_2 = len(Y2)

        # create test set
        X_test = X1[i*(count_1/n_fold_cv):(i+1)*(count_1/n_fold_cv)]
        X_test = np.vstack((X_test, X2[i*(count_2/n_fold_cv):(i+1)*(count_1/n_fold_cv)]))
        Y_test = Y1[i*(count_1/n_fold_cv):(i+1)*(count_1/n_fold_cv)]
        Y_test = np.hstack((Y_test, Y2[i*(count_2/n_fold_cv):(i+1)*(count_1/n_fold_cv)]))
        # create training set
        X_train = X1[:i*(count_1/n_fold_cv)]
        X_train = np.vstack((X_train, X2[:i*(count_2/n_fold_cv)]))
        X_train = np.vstack((X_train, X1[(i+1)*(count_1/n_fold_cv):]))
        X_train = np.vstack((X_train, X2[(i+1)*(count_2/n_fold_cv):]))
        Y_train = Y1[:i*(count_1/n_fold_cv)]
        Y_train = np.hstack((Y_train, Y2[:i*(count_2/n_fold_cv)]))
        Y_train = np.hstack((Y_train, Y1[(i+1)*(count_1/n_fold_cv):]))
        Y_train = np.hstack((Y_train, Y2[(i+1)*(count_2/n_fold_cv):]))

        return (X_train, Y_train, X_test, Y_test)

    def run(self, count=100, n_fold_cv=10):
        X1, Y1, X2, Y2 = self._generate_X_Y(self._get_data_from_db(count=count))
        X_train, Y_train, X_test, Y_test = self._split_X_Y(X1, Y1, X2, Y2, n_fold_cv, 0)

        self._logger.info('Training SVM...')
        self.svm.train(X_train, Y_train)

        Y_predict = self.svm.predict(X_test)
        correct = np.sum(Y_predict == Y_test)
        print "%d out of %d predictions correct" % (correct, len(Y_predict))


t = SVMTest('/home/tmarek/all/projects/dip/dip/data/articles/annotated.db')
t.run()
