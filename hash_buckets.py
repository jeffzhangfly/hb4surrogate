# -*- coding: utf-8 -*-
# !/usr/bin/env python

__author__ = 'jeff'
import copy


class HushBucket(object):
    __precisions = []
    __hashBuckets = []
    __valMax = []
    __valMin = []

    def __init__(self, precisions_array, val_max, val_min):
        self.__precisions = copy.deepcopy(precisions_array)
        self.__valMax = val_max
        self.__valMin = val_min
        # add the first hash, store original value
        self.__hashBuckets = []
        self.__hashBuckets.append({"precision": -1})
        # add hash for each precision
        for precision in self.__precisions:
            self.__hashBuckets.append({"precision": precision})

    def insert(self, position_ori, fitness):
        position = [self.__zoom_store(position_ori[i], i) for i in range(len(position_ori))]
        for ht in self.__hashBuckets:
            precision = ht["precision"]
            # original precision
            if precision == -1:
                ##################################################################
                # 1. key: convert vector to string
                # join each dimense to a string, splited by semicolon
                key = ''.join(str(v) + ';' for v in position)
                ht[key] = fitness

                ##################################################################

                ##################################################################
                # 2. key: convert vector to tuple
                # key = tuple(position)
                # ht[key] = fitness
                ##################################################################
            else:
                ##################################################################
                # zoom the scope of variables to [-1, 1]
                # x' = 2 * (x - x_min) / (x_max - x_min)  - 1            x' \in [-1, 1]
                # position_zoomed =

                ##################################################################

                ##################################################################
                # 1. key: convert vector to string
                # the original precision position of individual

                ori_position = ''.join(str(v) + ';' for v in position)
                # this precision position (as key)
                fmt = '%.' + '%d' % precision + 'f'
                key = ''.join(fmt % v + ';' for v in position)
                this_val = [ori_position, fitness]

                ##################################################################

                ##################################################################
                # 2. key: convert vector to tuple
                # the original precision position of individual
                # this precision position (as key)
                # ori_position = copy.deepcopy(position)
                # key = tuple(round(i, precision) for i in ori_position)
                # this_val = [ori_position, fitness]
                ##################################################################


                existed_val = []
                try:
                    existed_val = ht[key]  # get existed
                except KeyError:  # no existed value
                    existed_val = []

                #########################################################
                # needless if search original precision hast table before search other hast bucket
                #
                # # check if 'ori_position' in value
                # existed = False
                # for pos_fitness_pair in existed_val:
                #     if 0 == cmp(pos_fitness_pair[0], ori_position):
                #         existed = True
                #         break
                #
                # if not existed:
                #     existed_val.append(this_val)
                #     ht[key] = existed_val
                #########################################################

                existed_val.append(this_val)
                ht[key] = existed_val

    def get(self, key_ori, precision=-1):
        key_zoomed = [self.__zoom_store(key_ori[i], i) for i in range(len(key_ori))]
        for ht in self.__hashBuckets:
            prc = ht["precision"]
            if prc == precision:
                try:
                    if precision == -1:
                        # join each dimense to a string, splited by semicolon
                        key = ''.join(str(v) + ';' for v in key_zoomed)
                        val = ht[key]

                        # key = tuple(key)
                        # val = ht[key]
                    else:
                        # this precision position (as key)
                        fmt = '%.' + '%d' % precision + 'f'
                        key = ''.join(fmt % v + ';' for v in key_zoomed)
                        val = ht[key]

                        # key = tuple(round(i, precision) for i in key)
                        # val = ht[key]
                except KeyError:
                    val = None
        return val

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


    def get_buckets(self):
        return self.__hashBuckets


