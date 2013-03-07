#!/usr/bin/env python
# This is controll script for entire diploma thesis.

import argparse
import numpy as np

from src.svm.svm_test import *
from src.svm.svm_classifier import *
from src.svm.src.data import *
from src.svm.src.kernels import *

from src.bayes.bayesian_test import BayesianTest

from src.common.entry import Entry

# SVM
def svm_data(args):
    '''
    Functinon runs data regeneration according to input arguments
    '''
    t = SVMTest()
    t.regenerate_data(dbfile=args.path, count=args.count,
            max_token_size=args.max_token_size)

def svm_annealing(args):
    '''
    Function starts simulated annealing to find out ideal parameters for SVM
    classifier with given data.
    '''
    # twitter results: ((6.38946460577526, 33388.56022386515), 1.07)
    # article results:
    t = SVMTest()
    t.run_annealing(n_fold_cv=args.n_fold_cv, kernel=args.kernel)

def svm_test(args):
    '''
    Function starts test of SVM classifier with given data and classifier
    parameters. Only possible with previously created dataset - use svm_data()
    before using this!
    '''
    t = SVMTest()
    t.run(c=args.c, param=args.param, n_fold_cv=args.n_fold_cv,
            kernel=args.kernel)

def svm_create_model(args):
    '''
    Create SVM classifier model for separate classification
    '''
    # load data
    data = Data(dbfile=args.db_file, max_token_size=args.max_token_size)
    data.regenerate_X1_X2(99999)
    X, Y = data.get()

    # init kernel
    k = str2kernel[args.kernel](param=args.param)

    # crete classifier
    svm = SVM(kernel=k, C=args.c)
    svm.train(X, Y)
    # store model
    svm.store_model(args.model, data.get_token_list())

def svm_classify(args):
    '''
    Manually classify given text with some SVM model
    '''
    # load model and create classifier
    svm = SVM(kernel=None, C=None)
    token_list = svm.load_model(args.model)

    # convert text to vector X
    entry = Entry(id=None, guid=None, entry=args.text, language=None,
                    max_token_size=1)
    X = np.zeros((1,len(token_list)))
    for token in entry.get_token_all():
        if token.get_data_str() in token_list:
            X[0][token_list.index(token.get_data_str())] = 1

    # classify text
    print int(svm.predict(X)[0])

# BAYES
def bayes_test(args):
    '''
    Function starts test of bayesian classifier with given dataset and classifier
    parameters.
    '''
    features = eval(args.feats)
    if isinstance(features,dict):
        e = Entry(id=None, guid=None, entry=None, language=None)
        if not e.check_feats(features):
            return

    bt = BayesianTest(dbfile=args.db_file, low=args.low, high=args.high)
    bt.run(features=features, count=args.count, n_fold_cv=args.n_fold_cv)

def bayes_features(args):
    '''
    Function starts strats process of finding most suitable feature combination
    for selected dataset
    '''
    # twitter results: {'emoticon':1, 'sentence':1, 'url':4, 'tag':1, 'time':0,
    #                   'date':0, 'email':0}
    # process features
    # run
    bt = BayesianTest(dbfile=args.db_file, low=args.low, high=args.high)
    bt.get_best_features(count=args.count, n_fold_cv=args.n_fold_cv)


def parse_args():
    '''
    Function for parsing commandline arguments
    '''
    parser = argparse.ArgumentParser(description='''This project compares two
            commonly used classifiers - SVM and Bayesian classifier''')
    subparsers = parser.add_subparsers()

    # SVM
    parser_svm = subparsers.add_parser('svm',
            help='Operations with SVM classifier')
    subparsers_svm = parser_svm.add_subparsers()

    # SVM - data manipulation
    parser_svm_data = subparsers_svm.add_parser('data',
            help='Regenerate data')
    parser_svm_data.add_argument('--db_file', '-d', type=str, required=True,
            help='Database file with anotated data')
    parser_svm_data.add_argument('--count', '-c', default=9999, type=int,
            help='Number of processed entries')
    parser_svm_data.add_argument('--max_token_size','-t', default=1, type=int,
            help='Size of word n-tuples')
    parser_svm_data.set_defaults(func=svm_data)

    # SVM - anneailng process
    parser_svm_annealing = subparsers_svm.add_parser('annealing',
            help='Run simulated annealing to find optimal SVM parameters')
    parser_svm_annealing.add_argument('--n_fold_cv', '-n', type=int, default=5,
            help='Define number of used fold cross-validations')
    parser_svm_annealing.add_argument('--kernel', '-k', type=str, default='RBF',
            choices=['RBF', 'linear', 'polynomial'], help='select used kernel')
    parser_svm_annealing.set_defaults(func=svm_annealing)

    # SVM - separate classification
    parser_svm_classify = subparsers_svm.add_parser('classify',
            help='Classify text with given SVM classification model')
    parser_svm_classify.add_argument('--model', '-m', type=str, required=True,
            help='Path to pickled SVM model')
    parser_svm_classify.add_argument('--text', '-t', type=str, required=True,
            help='Input text')
    parser_svm_classify.set_defaults(func=svm_classify)

    # SVM - train svm model from annotated db file and store it to file
    parser_svm_model = subparsers_svm.add_parser('model',
            help='train SVM model from annotated db file and store it to file')
    parser_svm_model.add_argument('--model', '-m', type=str, required=True,
            help='Path to pickled SVM model')
    parser_svm_model.add_argument('--db_file', '-d', type=str, required=True,
            help='Annotated data file')
    parser_svm_model.add_argument('--max_token_size','-t', default=1, type=int,
            help='Size of word n-tuples')
    parser_svm_model.add_argument('--param', '-p', type=float, required=True,
            help='SVM kernel parameter')
    parser_svm_model.add_argument('--c', '-c', type=float, required=True,
            help='SVM classifier parameter C')
    parser_svm_model.add_argument('--kernel', '-k', type=str, default='RBF',
            choices=['RBF', 'linear', 'polynomial'], help='select used kernel')
    parser_svm_model.set_defaults(func=svm_create_model)

    # SVM - run tests
    parser_svm_test = subparsers_svm.add_parser('test',
            help='Run tests with given parameters')
    parser_svm_test.add_argument('--param', '-p', type=float, required=True,
            help='SVM classifier parameter gamma')
    parser_svm_test.add_argument('--c', '-c', type=float, required=True,
            help='SVM classifier parameter C')
    parser_svm_test.add_argument('--n_fold_cv', '-n', type=int, default=5,
            help='Defines number of used fold cross-validations')
    parser_svm_test.add_argument('--kernel', '-k', type=str, default='RBF',
            choices=['RBF', 'linear', 'polynomial'], help='select used kernel')
    parser_svm_test.set_defaults(func=svm_test)


    # BAYES
    parser_bayes = subparsers.add_parser('bayes',
            help='Operations with bayesian classifier')
    subparsers_bayes = parser_bayes.add_subparsers()

    # BAYES - run tests
    parser_bayes_test = subparsers_bayes.add_parser('test',
            help='Run tests with given parameters')
    parser_bayes_test.add_argument('--low', '-a', type=float, default=0.4,
            help='Low threshold for classifier')
    parser_bayes_test.add_argument('--high', '-b', type=float, default=0.6,
            help='High threshold for classifier')
    parser_bayes_test.add_argument('--count', '-c', type=int, default=5000,
            help='Count of precessed entries from DB')
    parser_bayes_test.add_argument('--db_file', '-d', type=str, required=True,
            help='Specify input database file for running tests')
    parser_bayes_test.add_argument('--n_fold_cv', '-n', type=int, default=5,
            help='Defines number of used fold cross-validations')
    parser_bayes_test.add_argument('--feats', '-f', type=str,
            default="{'emoticon':1, 'sentence':1, 'url':4, 'tag':1, 'time':0, \
                'date':0, 'email':0}",
            help='python array of possible features (see src/common/entry.py)')
    parser_bayes_test.set_defaults(func=bayes_test)

    # BAYES - run select features
    parser_bayes_test = subparsers_bayes.add_parser('features',
            help='Run process of finding ideal feature combinations')
    parser_bayes_test.add_argument('--low', '-a', type=float, default=0.4,
            help='Low threshold for classifier')
    parser_bayes_test.add_argument('--high', '-b', type=float, default=0.6,
            help='High threshold for classifier')
    parser_bayes_test.add_argument('--count', '-c', type=int, default=5000,
            help='Count of precessed entries from DB')
    parser_bayes_test.add_argument('--db_file', '-d', type=str, required=True,
            help='Specify input database file for running tests')
    parser_bayes_test.add_argument('--n_fold_cv', '-n', type=int, default=5,
            help='Defines number of used fold cross-validations')
    parser_bayes_test.set_defaults(func=bayes_features)
    # run argparse
    args = parser.parse_args()
    try:
        args.func(args)
    except argparse.ArgumentTypeError, ex:
        print ex



if __name__ == '__main__':
    logging.basicConfig(
            format='%(levelname)s:%(message)s',
            level=logging.DEBUG
        )

    # parse commandline arguments
    parse_args()
