#!/usr/bin/python3

import timeit
import numpy
import random

elements = 10000
numpyvector = numpy.ones(elements)
for n in xrange(elements):
    if random.random() > 0.5:
        numpyvector[n] = 0

print numpyvector

def testit(what):
    print("testing {}:".format(what))
    print(timeit.repeat(
        what,
        '''
import numpy
import math
from __main__ import  numpyvector'''))

print()
print('numpyvector =', repr(numpyvector))

# three methods for calculating RBF kernel dot products
testit('numpy.exp(-numpy.linalg.norm(numpyvector) ** 2 / (2 * (0.5 ** 2)))')
testit('numpy.exp(-math.fsum(x**2 for x in numpyvector.flat)/(2*(0.5**2)))')
testit('numpy.exp(-sum(x**2 for x in numpyvector.flat)/(2*(0.5**2)))')
