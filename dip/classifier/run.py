#!/usr/bin/env python

import sys
from optparse import OptionParser
from bayesian_classifier import Bayesian_Classifier

parser = OptionParser()
# classifier details
parser.add_option('-a', '--low', default='0.4', dest='low', help='low threshold for classifier', metavar='LOW')
parser.add_option('-b', '--high', default='0.6', dest='high', help='high threshold for classifier', metavar='HIGH')
parser.add_option('-c', '--count', default='100', dest='count', help='Count of precessed entries from DB', metavar='COUNT')
parser.add_option('-d', '--db-file', default=None, dest='db', help='Specify input database file for running tests')
parser.add_option('-r', '--run-tests', action='store_true', dest='tests', help='Run tests on pickled <File> with classified tweets')

(options, args) = parser.parse_args()

if not options.db:
    print 'specify db!!!!'
    sys.exit(1)

# create bayesian classifier
bcl = Bayesian_Classifier(dbfile=options.db, low=options.low, high=options.high)

if options.tests:
    bcl.run(count=options.count, n_fold_cv=10)
