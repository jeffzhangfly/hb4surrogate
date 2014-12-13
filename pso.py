# -*- coding: utf-8 -*-
# !/usr/bin/env python

__author__ = 'jeff'
import numpy as np
import time
import random
import hash_buckets as hb


class Pso(object):
    __c1 = 0.9  # para c1
    __c2 = 1.2  # para c2
    __dim = 0  # 问题维数
    __popNumber = 0  # 种群规模
    __maxGen = 0  # 最大进化代数
    # __maxNoImprove = 0  # 最大无进化代数
    __valMax = []  # 各维取值的最大值
    __valMin = []  # 各维取值的最小值
    __spdMax = 2.0  # 最大速度
    __spdMin = -2.0  # 最小速度
    __fitnessFun = None  # 适应值函数
    __breakCondition = 0  # 结束条件，fitness小于这个值阶数进化迭代

    __precisions = None
    __htbuck = None
    __min_neighbour_number = 9999

    __population = None
    __pop_speed = None

    __use_surrogate = False

    MAXFITNESS = 999999999

    # for statistic
    __fitnessEvalTimes = 0
    __directHitTimes = 0
    __approximateTimes = 0
    __approximateTimesPerBucket = []
    __approximated_better_than_gbest_then_calc = 0
    __timeInsert = 0
    __timeGetFromHash = 0
    __timeFitnessEval = 0

    def __init__(self, c1, c2, dim, pop_number, max_gen,
                 val_max, val_min,
                 fitness_fun, break_condition, use_surrogate,
                 prcs, min_neighbour_number):
        self.__c1 = c1
        self.__c2 = c2
        self.__dim = dim
        self.__popNumber = pop_number
        self.__maxGen = max_gen
        # self.maxNoImprove = max_no_improve

        # for statistic
        self.__fitnessEvalTimes = 0
        self.__directHitTimes = 0
        self.__approximateTimes = 0
        self.__approximateTimesPerBucket = []
        self.__approximated_better_than_gbest_then_calc = 0
        self.__timeInsert = 0
        self.__timeGetFromHash = 0
        self.__timeFitnessEval = 0

        if isinstance(val_max, list):
            self.__valMax = val_max
        else:
            self.__valMax = [val_max for i in range(self.__dim)]

        if isinstance(val_min, list):
            self.__valMin = val_min
        else:
            self.__valMin = [val_min for i in range(self.__dim)]

        spd_max = [((self.__valMax[i] - self.__valMin[i]) / 10) for i in range(len(self.__valMax))]
        if isinstance(spd_max, list):
            self.__spdMax = spd_max
        else:
            self.__spdMax = [spd_max for i in range(self.__dim)]
        self.__spdMin = [-1 * x for x in self.__spdMax]

        self.__fitnessFun = fitness_fun
        self.__breakCondition = break_condition
        self.__use_surrogate = use_surrogate

        if self.__use_surrogate:
            prcs.sort()
            self.__precisions = list(reversed([i for i in set(prcs)]))
            for i in range(len(self.__precisions)):
                self.__approximateTimesPerBucket.append(0)

            self.__htbuck = hb.HushBucket(self.__precisions, self.__valMax, self.__valMin)

            if isinstance(min_neighbour_number, list):
                self.__min_neighbour_number = min_neighbour_number
            else:
                self.__min_neighbour_number = [min_neighbour_number for i in range(self.__dim)]

    def statistics(self):
        return self.__fitnessEvalTimes, self.__directHitTimes, self.__approximateTimes, \
               self.__precisions, self.__approximateTimesPerBucket, \
               self.__approximated_better_than_gbest_then_calc, \
               self.__timeInsert, self.__timeGetFromHash, self.__timeFitnessEval

    def __init_population(self):
        # initialize population
        self.__population = np.random.uniform(self.__valMin, self.__valMax,
                                              (self.__popNumber, self.__dim))

    def iterate(self):
        self.__init_population()

        self.__pop_speed = np.random.uniform(self.__spdMin, self.__spdMax,
                                             (self.__popNumber, self.__dim))
        pbest_particle = self.__population.copy()
        pbest_fitness = np.random.uniform(self.MAXFITNESS - 1, self.MAXFITNESS, self.__popNumber)

        gbest_particle = np.empty(self.__dim, np.float)
        gbest_fitness = self.MAXFITNESS


        # calculate the fitness of initial population
        for i in range(self.__popNumber):
            self.__fitnessEvalTimes += 1
            start = time.time()
            fitness = self.__fitnessFun(self.__population[i])
            end = time.time()
            self.__timeFitnessEval += end - start
            if self.__use_surrogate:
                start = time.time()
                self.__htbuck.insert(self.__population[i], fitness)
                end = time.time()
                self.__timeInsert += end - start

            # check pbest
            if fitness < pbest_fitness[i]:
                pbest_fitness[i] = fitness
                for j in range(self.__dim):
                    pbest_particle[i][j] = self.__population[i][j]

            # check gbest
            if fitness < gbest_fitness:
                gbest_fitness = fitness
                for j in range(self.__dim):
                    gbest_particle[j] = self.__population[i][j]

            if gbest_fitness < self.__breakCondition:
                return gbest_fitness, gbest_particle

        # for statistic figure
        gbest_history = np.empty(self.__maxGen, np.float)
        pbest_avg_history = np.empty(self.__maxGen, np.float)

        # the first gbest and pbest average
        gbest_history[0] = gbest_fitness
        pbest_avg_history[0] = np.mean(pbest_fitness)

        for cur_gen in range(self.__maxGen - 1):
            # for each individual, get the next position
            for i in range(self.__popNumber):
                particle_next, particle_next_speed = self.__particle_step(
                    self.__pop_speed[i], self.__population[i],
                    pbest_particle[i], gbest_particle,
                    self.__maxGen, cur_gen)
                self.__pop_speed[i] = particle_next_speed
                self.__population[i] = particle_next

            # calculate the fitness
            for i in range(self.__popNumber):
                if self.__use_surrogate:
                    # check if this individual calculated
                    start = time.time()
                    sdata = self.__get_data_for_surrogate(self.__population[i])
                    end = time.time()
                    self.__timeGetFromHash += end - start

                    if sdata is None:  # not found anything
                        self.__fitnessEvalTimes += 1
                        start = time.time()
                        fitness = self.__fitnessFun(self.__population[i])
                        end = time.time()
                        self.__timeFitnessEval += end - start

                        start = time.time()
                        self.__htbuck.insert(self.__population[i], fitness)
                        end = time.time()
                        self.__timeInsert += end - start
                    elif (isinstance(sdata, float) or isinstance(sdata, int)):
                        fitness = sdata
                        self.__directHitTimes += 1
                    elif isinstance(sdata, list):
                        vars = []
                        for data in sdata:
                            vars.append(data[1])
                        fitness = self.__approximate_surrogate(vars)
                        if fitness < gbest_fitness:
                            # if approximate is better than gbest, fitness it
                            self.__fitnessEvalTimes += 1
                            self.__approximated_better_than_gbest_then_calc += 1
                            start = time.time()
                            fitness = self.__fitnessFun(self.__population[i])
                            end = time.time()
                            self.__timeFitnessEval += end - start

                            start = time.time()
                            self.__htbuck.insert(self.__population[i], fitness)
                            end = time.time()
                            self.__timeInsert += end - start
                        else:
                            self.__approximateTimes += 1
                    else:  # hash bucket failed
                        self.__fitnessEvalTimes += 1
                        start = time.time()
                        fitness = self.__fitnessFun(self.__population[i])
                        end = time.time()
                        self.__timeFitnessEval += end - start

                        start = time.time()
                        self.__htbuck.insert(self.__population[i], fitness)
                        end = time.time()
                        self.__timeInsert += end - start
                else:   # dont use hash
                    self.__fitnessEvalTimes += 1
                    start = time.time()
                    fitness = self.__fitnessFun(self.__population[i])
                    end = time.time()
                    self.__timeFitnessEval += end - start

                # check pbest
                if fitness < pbest_fitness[i]:
                    pbest_fitness[i] = fitness
                    for j in range(self.__dim):
                        pbest_particle[i][j] = self.__population[i][j]

                # check gbest
                if fitness < gbest_fitness:
                    gbest_fitness = fitness
                    for j in range(self.__dim):
                        gbest_particle[j] = self.__population[i][j]

                if gbest_fitness < self.__breakCondition:
                    return gbest_fitness

                # the first gbest and pbest average
                gbest_history[cur_gen + 1] = gbest_fitness
                pbest_avg_history[cur_gen + 1] = np.mean(pbest_fitness)

        return gbest_fitness, gbest_particle, gbest_history, pbest_avg_history

    def __particle_step(self, speed, individual, pbest, gbest,
                        max_generation, cur_generation):
        particle_next = np.empty(self.__dim, np.float)
        particle_next_speed = np.empty(self.__dim, np.float)

        w = self.__get_weight(max_generation, cur_generation)

        for i in range(self.__dim):
            r1 = random.uniform(0, 1)
            r2 = random.uniform(0, 1)

            particle_next_speed[i] = w * speed[i] + \
                                     self.__c1 * r1 * (pbest[i] - individual[i]) + \
                                     self.__c2 * r2 * (gbest[i] - individual[i])

            if particle_next_speed[i] > self.__spdMax[i]:
                particle_next_speed[i] = self.__spdMax[i]
            if particle_next_speed[i] < self.__spdMin[i]:
                particle_next_speed[i] = self.__spdMin[i]

            particle_next[i] = individual[i] + particle_next_speed[i]
            if particle_next[i] > self.__valMax[i]:
                particle_next[i] = self.__valMax[i]
            if particle_next[i] < self.__valMin[i]:
                particle_next[i] = self.__valMin[i]

        return particle_next, particle_next_speed


    def __get_weight(self, max_gen, cur_gen):
        return 0.9 - (float(cur_gen) / float(max_gen)) * 0.5


    def __get_data_for_surrogate(self, individual):
        ret = self.__htbuck.get(individual)
        if ret is not None:
            return ret

        i = 0
        for prc in self.__precisions:
            ret = self.__htbuck.get(individual, prc)
            if ret is not None:
                if len(ret) > self.__min_neighbour_number[i]:
                    ##################################################################
                    # DEBUG INFO
                    # if prc > 5:
                    #     print "##################################"
                    #     print "precision: " + str(prc)
                    #     print "Number: " + str(len(ret)) + ": " + str(ret[0])
                    #     print "Number: " + str(len(ret)) + ": " + str(ret[1])
                    #     print "Number: " + str(len(ret)) + ": " + str(ret[2])
                    #     print "##################################"
                    ##################################################################
                    self.__approximateTimesPerBucket[i] += 1
                    return ret
            i += 1

        return None


    def __approximate_surrogate(self, sdata):
        ret = np.mean(sdata)
        return ret

    # zoom ori value to store value in hash tabel
    # return value \in [-1, 1]
    def __zoom_store(self, val, dim_index):
        max = self.__valMax[dim_index]
        min = self.__valMin[dim_index]
        # x' = 2 * (x - x_min) / (x_max - x_min)  - 1       x' \in [-1, 1]
        return 2 * (val - min) / (max - min) - 1

    # zoom value \in [-1, 1] to original value
    def __zoom_original(self, val, dim_index):
        max = self.__valMax[dim_index]
        min = self.__valMin[dim_index]
        # x = (x' + 1) * (max - min) /2 + min
        return (val + 1) * (max - min) / 2 + min

    def get_hash_buckets_len(self, index):
        hts = self.__htbuck.get_buckets()
        return len(hts[index])


    def get_hash_buckets_precise(self, index):
        hts = self.__htbuck.get_buckets()
        ht = hts[index]
        return ht["precision"]
