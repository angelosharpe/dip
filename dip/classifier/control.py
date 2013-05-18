#!/usr/bin/env python
# This is controll script for entire diploma thesis.

import argparse
import numpy as np
import pp

from src.svm.svm_test import *
from src.svm.svm_classifier import *
from src.svm.src.data import *
from src.svm.src.kernels import *

from src.bayes.bayesian_test import BayesianTest
from src.bayes.bayesian_classifier import BayesianClassifier

from src.common.entry import Entry

# SVM
def svm_data(args):
    '''
    Functinon runs data regeneration according to input arguments
    '''
    t = SVMTest()
    t.regenerate_data(dbfile=args.db_file, count=args.count,
            max_token_size=args.max_token_size)

def svm_annealing(args):
    '''
    Function starts simulated annealing to find out ideal parameters for SVM
    classifier with given data.
    '''
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
    bt = BayesianTest(dbfile=args.db_file, max_token_size=args.max_token_size)

    if args.feats is not None:
        features = eval(args.feats)
        if isinstance(features,dict):
            e = Entry(id=None, guid=None, entry=None, language=None)
            if not e.check_feats(features):
                print 'Incorrect format of feature dictionary'
                return
    else:
        features = bt.get_best_features(count=args.count, n_fold_cv=args.n_fold_cv)

    bt.run(features=features, count=args.count, n_fold_cv=args.n_fold_cv)

def bayes_features(args):
    '''
    Function starts strats process of finding most suitable feature combination
    for selected dataset
    '''
    # process features
    # run
    bt = BayesianTest(dbfile=args.db_file, max_token_size=args.max_token_size)
    bt.get_best_features(count=args.count, n_fold_cv=args.n_fold_cv)

def bayes_generate_model(args):
    '''
    Function creates model for bayesian classifier
    '''
    bt = BayesianTest(dbfile=args.db_file, max_token_size=args.max_token_size)

    if args.feats is not None:
        features = eval(args.feats)
        if isinstance(features,dict):
            e = Entry(id=None, guid=None, entry=None, language=None)
            if not e.check_feats(features):
                print 'Incorrect format of feature dictionary'
                return
    else:
        features = bt.get_best_features(count=args.count, n_fold_cv=args.n_fold_cv)

    bt.create_model(args.model, used_features=features, count=args.count)


def bayes_classify(args):
    '''
    Manually classify given text with some bayesian model
    '''
    bcl = BayesianClassifier()
    bcl.load_word_dict(args.model)
    print bcl.classify(text=args.text, language='en', features=bcl.word_dict.words['features'])


# COMMON
def _print_results(t,tp, fp, tn, fn, u=None, c=None):
    '''
    method prints results from of testing
    @param t: type (svm or bayes)
    @param tp: true positive reults
    @param fp: false positive results
    @param tn: true negative results
    @param fn: false negative results
    @param u: unknown results
    @param c: corelation results
    '''
    precision = tp / ((tp + fp) + 0.0000000000001)
    recall = tp / ((tp + fn) + 0.0000000000001)
    acc = (tp + tn) / ((tp + tn + fn + fn) + 0.0000000000001)
    f_measure = 2 * ((precision * recall)/((precision + recall) + 0.0000000000001))
    ret = '##################################\n'
    ret += '########## ' + str(t) + ' restults ##########\n'
    ret += '##################################\n'
    ret += 'True positive = ' + str(tp) + '\n'
    ret += 'True negative = ' + str(tn) + '\n'
    ret += 'False positive = ' + str(fp) + '\n'
    ret += 'False negative = ' + str(fn) + '\n'
    if u is not None:
        ret += 'Unknown = ' + str(u) + '\n'
    ret += 'Precision = ' + str(precision) + '\n'
    ret += 'Recall = ' + str(recall) + '\n'
    ret += 'Accuracy = ' + str(acc) + '\n'
    ret += 'F-measure = ' + str(f_measure) + '\n'
    if c is not None:
        ret += 'Corelation = ' + str(c) + '\n'
    ret += '##################################\n'
    print ret



def _thread_svm(db_file, count, max_token_size, n_fold_cv, kernel):
    from src.svm.svm_test import SVMTest
    t = SVMTest()
    # load data
    t.regenerate_data(dbfile=db_file, count=count,
            max_token_size=max_token_size)
    # run simulated annealing
    state, energy = t.run_annealing(n_fold_cv=n_fold_cv, kernel=kernel)
    # run test with optimal parameters
    result = t.run(c=state[1], param=state[0], n_fold_cv=n_fold_cv,
            kernel=kernel)
    # return results
    return {'type':'svm', 'result': result, 'state':state, 'emergy':energy}

def _thread_bayes(db_file, count, n_fold_cv, max_token_size):
    from src.bayes.bayesian_test import BayesianTest
    bt = BayesianTest(dbfile=db_file, max_token_size=max_token_size)
    # run feature selection
    features = bt.get_best_features(count=count, n_fold_cv=n_fold_cv)
    # run test with best features
    result = bt.run(features=features, count=count, n_fold_cv=n_fold_cv)
    # return results
    return {'type':'bayes', 'result': result, 'features':features}

def common_run(args):
    '''
    Run comparison of SVM and bayesian classifiers
    '''
    # setup job server
    job_server = pp.Server()
    job_server.set_ncpus()
    # create jobs
    jobs = []
    jobs.append(job_server.submit(_thread_svm, (args.db_file, args.count,
        args.max_token_size, args.n_fold_cv, args.kernel)))
    jobs.append(job_server.submit(_thread_bayes, (args.db_file, args.count,
        args.n_fold_cv, args.max_token_size)))
    result = [job() for job in jobs]
    # print result
    print result
    for r in result:
        if r['type'] == 'bayes':
            _print_results(t=r['type'], tp=r['result']['true_positive'],
                fp=r['result']['false_positive'], tn=r['result']['true_negative'],
                fn=r['result']['false_negative'], u=r['result']['unknown'],
                c=r['result']['corelation'])
            print r['features']
        else:
            _print_results(t=r['type'], tp=r['result']['true_positive'],
                fp=r['result']['false_positive'], tn=r['result']['true_negative'],
                fn=r['result']['false_negative'])

def parse_args():
    '''
    Function for parsing commandline arguments
    '''
    parser = argparse.ArgumentParser(description='''Tento program ma za cil porovnat dva casto pouzivane klasifikatory -- SVM a Bayesovsky klasifikator. V obou implementovanych klasifikatorech je kladen duraz na vyber optimalnich priznaku ziskavanych z textu. Jsou ziskavany dva typy priznaku  -- textove priznaky a specialni priznaky.  Program vznikl jako implementacni cast diplomove prace ve ktere taky zkoumame vliv techto priznaku na klasifikacni schopnosti klasifikatoru.''')
    subparsers = parser.add_subparsers()

    # SVM
    parser_svm = subparsers.add_parser('svm',
            help='Operace s SVM klasifikatorem.')
    subparsers_svm = parser_svm.add_subparsers()

    # SVM - data manipulation
    parser_svm_data = subparsers_svm.add_parser('data',
            help='''Tato operace vygeneruje vnitrni reprezentaci datove sady se kterou klasifikator SVM nasledne pracuje.
            Tato operace je provadena samostatne kvuli jeji casove narocnosti.''')
    parser_svm_data.add_argument('--db_file', '-d', type=str, required=True,
            help='Cesta k databazovemu souboru s anotovanymi daty')
    parser_svm_data.add_argument('--count', '-c', default=9999, type=int,
            help='Pocet pouzitych zaznamu z databazoveho souboru')
    parser_svm_data.add_argument('--max_token_size','-t', default=1, type=int,
            help='Maximalni delka n-tic textovych priznaku')
    parser_svm_data.set_defaults(func=svm_data)

    # SVM - anneailng process
    parser_svm_annealing = subparsers_svm.add_parser('annealing',
            help='Spusteni procesu simulovaneho zihani hledajiciho optimalni nastaveni volnych parametru SVM klasifikatoru a jaderne funkce. Pred spustenim tohoto procesu by mela byt vygenerovana data.')
    parser_svm_annealing.add_argument('--n_fold_cv', '-n', type=int, default=5,
            help='Pocet iteraci n-nasobne krizove validace')
    parser_svm_annealing.add_argument('--kernel', '-k', type=str, default='RBF',
            choices=['RBF', 'linear', 'polynomial'], help='Vyber jaderne funkce')
    parser_svm_annealing.set_defaults(func=svm_annealing)

    # SVM - train svm model from annotated db file and store it to file
    parser_svm_model = subparsers_svm.add_parser('model',
            help='Vytvori klasifikacni model pro SVM klasifikator ktery umozni v budoucnosti klasifikovat bez nutnosti opetovneho uceni.')
    parser_svm_model.add_argument('--model', '-m', type=str, required=True,
            help='Soubor nove vytvoreneho modelu')
    parser_svm_model.add_argument('--db_file', '-d', type=str, required=True,
            help='Cesta k databazovemu souboru s anotovanymi daty')
    parser_svm_model.add_argument('--max_token_size','-t', default=1, type=int,
            help='Maximalni delka n-tic textovych priznaku')
    parser_svm_model.add_argument('--param', '-p', type=float, required=True,
            help='Parametr jaderne funkce')
    parser_svm_model.add_argument('--c', '-c', type=float, required=True,
            help='Parametr C SVM klasifikatoru')
    parser_svm_model.add_argument('--kernel', '-k', type=str, default='RBF',
            choices=['RBF', 'linear', 'polynomial'], help='Vyber jaderne funkce')
    parser_svm_model.set_defaults(func=svm_create_model)


    # SVM - separate classification
    parser_svm_classify = subparsers_svm.add_parser('classify',
            help='Klasifikuje dany vstupni tex za pouziti modelu.')
    parser_svm_classify.add_argument('--model', '-m', type=str, required=True,
            help='Soubor klasifikacniho modelu')
    parser_svm_classify.add_argument('--text', '-t', type=str, required=True,
            help='Vstupni text')
    parser_svm_classify.set_defaults(func=svm_classify)

    # SVM - run tests
    parser_svm_test = subparsers_svm.add_parser('test',
            help='Spusti test SVM klasifikatoru s danymi parametry')
    parser_svm_test.add_argument('--param', '-p', type=float, required=True,
            help='Parametr jaderne funkce')
    parser_svm_test.add_argument('--c', '-c', type=float, required=True,
            help='Parametr C SVM klasifikatoru')
    parser_svm_test.add_argument('--n_fold_cv', '-n', type=int, default=5,
            help='Pocet iteraci n-nasobne krizove validace')
    parser_svm_test.add_argument('--kernel', '-k', type=str, default='RBF',
            choices=['RBF', 'linear', 'polynomial'], help='Vyber jaderne funkce')
    parser_svm_test.set_defaults(func=svm_test)


    # BAYES
    parser_bayes = subparsers.add_parser('bayes',
            help='Operace s Bayesovskym klasifikatorem ')
    subparsers_bayes = parser_bayes.add_subparsers()

    # BAYES - run tests
    parser_bayes_test = subparsers_bayes.add_parser('test',
            help='Spusti test Bayesovskeho klasifikatoru s danymi parametry.')
    parser_bayes_test.add_argument('--count', '-c', type=int, default=5000,
            help='Pocet pouzitych zaznamu z databazoveho souboru')
    parser_bayes_test.add_argument('--db_file', '-d', type=str, required=True,
            help='Cesta k databazovemu souboru s anotovanymi daty')
    parser_bayes_test.add_argument('--n_fold_cv', '-n', type=int, default=5,
            help='Pocet iteraci n-nasobne krizove validace')
    parser_bayes_test.add_argument('--max_token_size','-t', default=1, type=int,
            help='Maximalni delka n-tic textovych priznaku')
    parser_bayes_test.add_argument('--feats', '-f', type=str,
            default=None,
            help='Slovnik pythonu definujici uzite spec. priznaky, pokud neni definovan, je spusteno hledani optimalnich spec. priznaku')
    parser_bayes_test.set_defaults(func=bayes_test)

    # BAYES - run select features
    parser_bayes_feature = subparsers_bayes.add_parser('features',
            help='Spusti proces vyhledavani optimalnich specialnich priznaku.')
    parser_bayes_feature.add_argument('--count', '-c', type=int, default=5000,
            help='Pocet pouzitych zaznamu z databazoveho souboru')
    parser_bayes_feature.add_argument('--db_file', '-d', type=str, required=True,
            help='Cesta k databazovemu souboru s anotovanymi daty')
    parser_bayes_feature.add_argument('--n_fold_cv', '-n', type=int, default=5,
            help='Pocet iteraci n-nasobne krizove validace')
    parser_bayes_feature.add_argument('--max_token_size','-t', default=1, type=int,
            help='Maximalni delka n-tic textovych priznaku')
    parser_bayes_feature.set_defaults(func=bayes_features)

    # BAYES - generate model
    parser_bayes_model = subparsers_bayes.add_parser('model',
            help='Vytvori klasifikacni model pro Bayesovsky klasifikator')
    parser_bayes_model.add_argument('--model', '-m', type=str, required=True,
            help='Soubor nove vytvoreneho modelu')
    parser_bayes_model.add_argument('--db_file', '-d', type=str, required=True,
            help='Cesta k databazovemu souboru s anotovanymi daty')
    parser_bayes_model.add_argument('--count', '-c', type=int, default=5000,
            help='Pocet pouzitych zaznamu z databazoveho souboru')
    parser_bayes_model.add_argument('--n_fold_cv', '-n', type=int, default=5,
            help='Pocet iteraci n-nasobne krizove validace')
    parser_bayes_model.add_argument('--max_token_size','-t', default=1, type=int,
            help='Maximalni delka n-tic textovych priznaku')
    parser_bayes_model.add_argument('--feats', '-f', type=str,
            default=None,
            help='Slovnik pythonu definujici uzite spec. priznaky, pokud neni definovan, je spusteno hledani optimalnich spec. priznaku')
    parser_bayes_model.set_defaults(func=bayes_generate_model)

    # BAYES - classify
    parser_bayes_classify = subparsers_bayes.add_parser('classify',
            help='Klasifikuje dany vstupni tex za pouziti modelu.')
    parser_bayes_classify.add_argument('--model', '-m', type=str, required=True,
            help='Soubor klasifikacniho modelu')
    parser_bayes_classify.add_argument('--text', '-t', type=str, required=True,
            help='Vstupni text')

    parser_bayes_classify.set_defaults(func=bayes_classify)

    # COMMON - compare svm and bayesian classifiers
    parser_common = subparsers.add_parser('common',
            help='''Porovnani Bayesovskeho a SVM klasifikatoru. Tento proces muze v zavislosti na zvolenych parametrech trvat velmi dlouho a zabrat velke mnozstvi pameti.''')
    parser_common.add_argument('--db_file', '-d', type=str, required=True,
            help='Cesta k databazovemu souboru s anotovanymi daty')
    parser_common.add_argument('--n_fold_cv', '-n', type=int, default=5,
            help='Pocet iteraci n-nasobne krizove validace')
    parser_common.add_argument('--count', '-c', type=int, default=5000,
            help='Pocet pouzitych zaznamu z databazoveho souboru')
    parser_common.add_argument('--max_token_size','-t', default=1, type=int,
            help='Maximalni delka n-tic textovych priznaku')
    # SVM params
    parser_common.add_argument('--kernel', '-k', type=str, default='RBF',
            choices=['RBF', 'linear', 'polynomial'],
            help='SVM: Vyber jaderne funkce')
    parser_common.set_defaults(func=common_run)

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
