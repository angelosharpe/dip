#!/usr/bin/env python

import numpy as np
import random
import math
import sys
from scipy.stats import chi2

from data import Data
from ..svm_classifier import SVM
from kernels import *

class Annealing():
    #parameters bounds
    P_MIN = 0.0001
    P_MAX = 30
    C_MIN = 0.0001
    C_MAX = 35000

    def __init__(self, kernel=None, init_temp=sys.maxint, D=2, iter_limit=1000):
        random.seed()
        self.iter_limit = iter_limit

        # svm
        if not kernel:
            self.kernel = LinearKernel()
        else:
            self.kernel = kernel
        self.svm = None

        # get data
        self.data = Data(dbfile=None)
        self.data.load_X1_X2()

        self.temp = float(init_temp)
        self.D = D
        self.chi2 = chi2.ppf(0.99, D)
        self.denom = (1.0/(0.9**(D/2.0)))-1

        self.state, self.energy = self._init_state()
        self.best_energy = self.energy
        self.best_state = self.state

    def _init_state(self):
        '''
        Generates random initial state
        @return tuple representing best initial state -- (param, C)
        '''
        state = (random.uniform(self.P_MIN, self.P_MAX),
                random.uniform(self.C_MIN, self.C_MAX))
        energy = self._get_energy(state)
        return (state, energy)

    def _generate_neighbor(self):
        '''
        Generate neigbor point. Generate direction vector and find random point
        on this vector.
        @return tuple representing neighbor -- (param, C)
        '''
        # generate direction point in a circle around current state
        while True:
            direction = (random.uniform(self.state[0]-1, self.state[0]+1),
                         random.uniform(self.state[1]-1, self.state[1]+1))
            v = (direction[0]-self.state[0], direction[1]-self.state[1])
            length = math.sqrt(v[0]**2 + v[1]**2)
            if length < 1:
                break
        # left
        if v[0] < 0:
            x_mult = (self.state[0] / v[0]) * -1
        # right
        else:
            x_mult = (self.P_MAX - self.state[0]) / v[0]
        # down
        if v[1] < 0:
            y_mult = (self.state[1] / v[1]) * -1
        # up
        else:
            y_mult = (self.C_MAX - self.state[1]) / v[1]
        # get maximal possible multiplier
        mult = random.uniform(0, min(x_mult, y_mult))
        return (self.state[0] + (mult * v[0]), self.state[1] + (mult * v[1]))

    def _get_energy(self, state):
        '''
        This method generates calculates energy of some state of svm classifier.
        @return tuple of energies, first one has higher priority than second one
        '''
        print 'calculating energy for state {0}'.format(state)
        # get data (only for iteration 0 of 10fcv)
        X1, Y1, X2, Y2 = self.data.get(0)
        # apply state
        self.kernel.change_param(state[0])
        self.svm = SVM(kernel=self.kernel, C=state[1])
        self.svm.train(X1, Y1)
        if self.svm.model_exists:
            Y_predict = self.svm.predict(X2)
            correct = np.sum(Y_predict == Y2)
            print '{0} out of {1} predictions correct'.format(correct,
                    len(Y_predict))
            # energy is calculated as percentage of incorrectly classified vectors and
            # percentage of used SV ^ 0.5  -- trying to minimize vectors
            incorrect_energy = 1 - (float(correct)/len(Y_predict))
            sv_energy = (float(self.svm.lm_count)/self.svm.all_lm_count)**0.5
            return incorrect_energy + sv_energy
        else:
            return sys.maxint

    def _jump_probability(self, neighbor_energy):
        '''
        Method returns probability of jumping to neighbor state. If neighbor
        state has lower energy, than actual state, jump is done right away,
        If the element has higher energy, there is still slight possibility of
        jump.
        @param neighbor_energy Energy of neighbor state
        '''
        # priority is to mimimize error rate, then to minimize number of SV
        difference = neighbor_energy - self.energy
        if difference < 0:
            return 1.0
        else:
            return math.e**(-difference/self.temp)

    def run(self):
        '''
        Run simulated annealing!!!
        '''
        iteration = 0
        while self.iter_limit > iteration:
            iteration += 1
            print '==current temperature is {0}, iteration:{1}=='.format(
                    self.temp, iteration)
            print 'best state: {}, best_energy: {}'.format(self.best_state, self.best_energy)
            print 'current state is:', self.state
            print 'current energy is:', self.energy
            neighbor = self._generate_neighbor()
            print 'new neighbor:', neighbor
            neighbor_energy = self._get_energy(neighbor)
            print 'neighbor energy:', neighbor_energy
            prob = self._jump_probability(neighbor_energy)
            r = random.random()
            print 'if {0} > {1} then use new'.format(prob, r)
            if prob > r:
                self.state = neighbor
                self.energy = neighbor_energy
                if self.energy < self.best_energy:
                    energy_temp = self.energy + (
                            (self.best_energy-self.energy) / self.denom)
                    self.temp = 2.0 * (energy_temp-self.energy) / self.chi2
                    self.best_state = self.state
                    self.best_energy = self.energy
                    print 'best_state={0}, best_energy={1}'.format(self.state,
                            self.energy)
                # if energies are the same, try to find states with minimal C and gamma
                if self.energy == self.best_energy:
                    best_energy = (self.best_state[0]-self.P_MIN)/(self.P_MAX-self.P_MIN) + \
                                  (self.best_state[1]-self.C_MIN)/(self.C_MAX-self.C_MIN)
                    new_energy = (self.state[0]-self.P_MIN)/(self.P_MAX-self.P_MIN) + \
                                 (self.state[1]-self.C_MIN)/(self.C_MAX-self.C_MIN)
                    if new_energy < best_energy:
                        self.best_state = self.state
                        self.best_energy = self.energy
                        print 'best_state={0}, best_energy={1}'.format(self.state,
                                self.energy)
        return self.best_state, self.best_energy
