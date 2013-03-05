#!/usr/bin/env python
# This is controll script for entire diploma thesis.

import argparse

from src.svm.svm_test import *
from src.bayes.bayesian_test import BayesianTest

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
    t.run_annealing(args.n_fold_cv)

def svm_test(args):
    '''
    Function starts test of SVM classifier with given data and classifier
    parameters. Only possible with previously created dataset - use svm_data()
    before using this!
    '''
    t = SVMTest()
    t.run(c=args.c, gamma=args.gamma, n_fold_cv=args.n_fold_cv)

def svm_classify(args):
    '''
    Manually classify given text with some SVM model
    '''
    # TODO:
    pass

# BAYES
def bayes_test(args):
    '''
    Function starts test of bayesian classifier with given dataset and classifier
    parameters.
    '''
    bt = BayesianTest(dbfile=args.db_file, low=args.low, high=args.high)
    bt.run(count=args.count, n_fold_cv=args.n_fold_cv)


def parse_args():
    '''
    Function for parsing commandline arguments
    '''
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # SVM
    parser_svm = subparsers.add_parser('svm')
    subparsers_svm = parser_svm.add_subparsers()

    # SVM - data manipulation
    parser_svm_data = subparsers_svm.add_parser('data')
    parser_svm_data.add_argument('--db_file', '-d', type=str, required=True)
    parser_svm_data.add_argument('--count', '-c', default=9999, type=int)
    parser_svm_data.add_argument('--max_token_size','-t', default=1, type=int)
    parser_svm_data.set_defaults(func=svm_data)

    # SVM - anneailng process
    parser_svm_annealing = subparsers_svm.add_parser('annealing')
    parser_svm_annealing.add_argument('--n_fold_cv', '-n', type=int, default=5,
            help='Defines number of used fold cross-validations')
    parser_svm_annealing.set_defaults(func=svm_annealing)

    # SVM - separate classification
    parser_svm_classify = subparsers_svm.add_parser('classify')
    parser_svm_classify.add_argument('--data', '-d', type=str, required=True,
            help='Path to trained SVM model')
    parser_svm_classify.add_argument('--text', '-t', type=str, required=True,
            help='Input text')
    parser_svm_classify.add_argument('--gamma', '-g', type=float, required=True,
            help='SVM classifier parameter gamma')
    parser_svm_classify.add_argument('--c', '-c', type=float, required=True,
            help='SVM classifier parameter C')
    parser_svm_classify.set_defaults(func=svm_classify)

    # SVM - run tests
    parser_svm_test = subparsers_svm.add_parser('test')
    parser_svm_test.add_argument('--gamma', '-g', type=float, required=True,
            help='SVM classifier parameter gamma')
    parser_svm_test.add_argument('--c', '-c', type=float, required=True,
            help='SVM classifier parameter C')
    parser_svm_test.add_argument('--n_fold_cv', '-n', type=int, default=5,
            help='Defines number of used fold cross-validations')
    parser_svm_test.set_defaults(func=svm_test)


    # BAYES
    parser_bayes = subparsers.add_parser('bayes')
    subparsers_bayes = parser_bayes.add_subparsers()

    # BAYES - run tests
    parser_bayes_test = subparsers_bayes.add_parser('test')
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
    parser_bayes_test.set_defaults(func=bayes_test)

    # run argparse
    args = parser.parse_args()
    try:
        args.func(args)
    except argparse.ArgumentTypeError, ex:
        ssm_parser.parser.error(ex)



if __name__ == '__main__':
    logging.basicConfig(
            format='%(levelname)s:%(message)s',
            level=logging.DEBUG
        )

    # parse commandline arguments
    parse_args()
