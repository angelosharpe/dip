#!/usr/bin/env python

import numpy as np
import random
import math
import sys
import logging
import pp

from data import Data
from ..svm_classifier import SVM
from kernels import *

def _thread_get_energy(svm, data, state, i, n_fold_cv):
    '''
    This function calculates energy of given state. This is a separate function
    because of parallelisation restrictions in python.
    @param svm used support vector machine object
    @param data tuple containing X1, X2, Y1 and Y2
    @param state tuple containing gamma and C
    @param i i-th step of n-fold cross-validation
    @param n_fold_cv specification of cross validation
    '''
    import numpy as np
    # get data (only for iteration 0 of 10fcv)
    X1, Y1, X2, Y2 = data
    svm.train(X1, Y1)
    if svm.model_exists:
        print 'n-fold c-v: iteration {0} of {1}'.format(i + 1, n_fold_cv)
        Y_predict = svm.predict(X2)
        correct = np.sum(Y_predict == Y2)
        print '{0} out of {1} predictions correct'.format(correct, len(Y_predict))
        print '{0} SV out of {1} vectors used'.format(svm.lm_count, svm.all_lm_count)
        # energy is calculated as percentage of incorrectly classified vectors
        # and percentage of used SV ^ 0.5  -- trying to minimize vectors
        incorrect_energy = 1 - (float(correct)/len(Y_predict))
        sv_energy = (float(svm.lm_count)/svm.all_lm_count)**0.5
        energy = (incorrect_energy + sv_energy)
        print 'current energy = {0}'.format(energy)
        return energy / n_fold_cv
    else:
        return sys.maxint

class Annealing():
    #parameters bounds
    P_MIN = 0.0001
    P_MAX = 30
    C_MIN = 0.0001
    C_MAX = 35000

    def __init__(self, kernel=None, init_temp=sys.maxint, iter_limit=1000,
                 n_fold_cv=None, max_token_size=1):
        # add and setup logger
        self._logger = logging.getLogger()
        self._logger.setLevel(logging.DEBUG)

        # parallelisation
        self.job_server = pp.Server()
        self.job_server.set_ncpus()

        random.seed()
        self.iter_limit = iter_limit

        # svm krenel init
        if not kernel:
            self.kernel = LinearKernel()
        else:
            self.kernel = kernel
        self.svm = None

        # get data, set cross validation
        if not n_fold_cv or n_fold_cv == 1:
            self.n_fold_cv = 1
            self.data = Data(dbfile=None)
        else:
            self.n_fold_cv = n_fold_cv
            self.data = Data(dbfile=None, n_fold_cv=n_fold_cv,
                    max_token_size=max_token_size)
        self.data.load_X1_X2()

        self.temp = float(init_temp)

        self.state, self.energy = self._init_state(2)
        self.best_energy = self.energy
        self.best_state = self.state

    def _init_state(self, matrix_size=5):
        '''
        Generates random initial state
        @return tuple representing best initial state -- (param, C)
        '''
        best_state = ()
        best_energy = None

        X1, Y1, X2, Y2 = self.data.get(0)

        self._logger.info('generating inital feasible point...')
        step_C = (self.C_MAX - self.C_MIN) / float(matrix_size - 1)
        step_P = (self.P_MAX - self.P_MIN) / float(matrix_size - 1)
        for i in xrange(matrix_size):
            for j in xrange(matrix_size):
                state = ((step_P * j) + self.P_MIN, (step_C * i) + self.C_MIN)
                energy = self._get_energy(state)
                if energy < best_energy or not best_energy:
                    best_energy = energy
                    best_state = state
                    self._logger.info('best_state:{0}, best_energy:{1}'
                        .format(best_state, best_energy))
        return (best_state, best_energy)

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
        If n-fold cross validation is enabled, energy is calclated using it.
        @param state tuple containing gamma and C
        @return tuple of energies, first one has higher priority than second one
        '''
        # apply state
        self.kernel.change_param(state[0])
        self.svm = SVM(kernel=self.kernel, C=state[1], silent=True)
        self._logger.info('calculating energy for state {0}'.format(state))
        jobs = []
        for i in xrange(self.n_fold_cv):
            data = self.data.get(i)
            jobs.append(self.job_server.submit(_thread_get_energy,
                (self.svm, data, state, i, self.n_fold_cv)))
        avg_energy = sum([job() for job in jobs])
        self._logger.debug('average energy = {0}'.format(avg_energy))
        return avg_energy

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
        if difference <= 0:
            return 1.0
        else:
            return math.e**(-difference/self.temp)

    def _get_temperature(self, iteration):
        '''
        Returns temperature according to the state of annealing process.
        @param iteration current iteration
        '''
        return  (1 - (float(iteration) / self.iter_limit)) * self.iter_limit

    def run(self):
        '''
        Run simulated annealing!!!
        '''
        iteration = 0
        while self.iter_limit > iteration:
            iteration += 1
            self.temp = self._get_temperature(iteration)
            self._logger.info('==current temperature is {0}, iteration:{1}=='
                    .format(self.temp, iteration))
            self._logger.info('best state: {}, best_energy: {}'
                    .format(self.best_state, self.best_energy))
            self._logger.info('current state is:{0}'.format(self.state))
            self._logger.info('current energy is:{0}'.format(self.energy))
            neighbor = self._generate_neighbor()
            if neighbor == self.state:
                self._logger.info('neighbor is same as previous state, skipping...')
                continue
            self._logger.info('new neighbor: {0}'.format(neighbor))
            neighbor_energy = self._get_energy(neighbor)
            self._logger.info('neighbor energy: {0}'.format(neighbor_energy))
            prob = self._jump_probability(neighbor_energy)
            r = random.random()
            self._logger.debug('if {0} > {1} then use new'.format(prob, r))
            if prob > r:
                self.state = neighbor
                self.energy = neighbor_energy
                if self.energy < self.best_energy:
                    self.best_state = self.state
                    self.best_energy = self.energy
                    self._logger.info('best_state={0}, best_energy={1}'
                            .format(self.state, self.energy))
                # if energies are the same, try to find states with minimal C and gamma
                if self.energy == self.best_energy:
                    best_energy = (self.best_state[0]-self.P_MIN)/(self.P_MAX-self.P_MIN) + \
                                  (self.best_state[1]-self.C_MIN)/(self.C_MAX-self.C_MIN)
                    new_energy = (self.state[0]-self.P_MIN)/(self.P_MAX-self.P_MIN) + \
                                 (self.state[1]-self.C_MIN)/(self.C_MAX-self.C_MIN)
                    if new_energy < best_energy:
                        self.best_state = self.state
                        self.best_energy = self.energy
                        self._logger.info('best_state={0}, best_energy={1}'
                                .format(self.state, self.energy))
        return self.best_state, self.best_energy
