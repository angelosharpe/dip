#!/usr/bin/env python

import numpy as np

class Kernel():
    '''
    Kernel class
    '''
    def __init__(self, param=None):
        self.param = param
    def change_param(self, new_param):
        self.param = new_param


class LinearKernel(Kernel):
    '''
    Linear kernel class, for calculating dotproduct in input feature space
    '''
    def __call__(self, x1, x2):
        '''
        given points x1 and x2 calculates dot product in input feature space
        @param x1: point x1
        @param x2: point x2
        @return: dot product
        '''
        return np.dot(x1, x2)

class PolynomialKernel(Kernel):
    '''
    Polynomial kernel class, for calculating dotproduct in transformed feature space
    '''
    def __call__(self, x1, x2):
        '''
        given points x1 and x2 calculates dot product in transformed feature space
        @param x1: point x1
        @param x2: point x2
        @return: dot product
        '''
        return (1 + np.dot(x1, x2)) ** self.param


class RBFKernel(Kernel):
    '''
    RBF kernel class, for calculating dotproduct in transformed feature space
    '''
    def __call__(self, x1, x2):
        '''
        given points x1 and x2 calculates dot product in transformed feature space
        @param x1: point x1
        @param x2: point x2
        @return: dot product
        '''
        return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * (self.param ** 2)))

str2kernel = {'RBF':RBFKernel, 'linear':LinearKernel, 'polynomial':PolynomialKernel}
