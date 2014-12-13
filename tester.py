# -*- coding: utf-8 -*-
# !/usr/bin/env python

__author__ = 'jeff'

import unittest as ut
import hash_buckets as hb


class HashBucketTest(ut.TestCase):
    # def setUp(self):

    # def tearDown(self):

    def testInsertAndGet(self):
        precisions = [6, 4, 2]
        htbuck = hb.HushBucket(precisions)

        vector1 = [3.12345678, 3.11111111, 3.66666666]
        vector2 = [2.12345678, 2.11111111, 2.66666666]
        vector3 = [1.12345678, 1.11111111, 1.66666666]
        fitness1 = 9.99
        fitness2 = 6.66
        fitness3 = 1.36

        vector4 = [1.12349999, 1.111119876, 1.6666999999]
        vector5 = [1.1234569999, 1.111111222211, 1.666666999966]
        fitness4 = 4.0
        fitness5 = 6.0

        htbuck.insert(vector1, fitness1)
        htbuck.insert(vector2, fitness2)
        htbuck.insert(vector3, fitness3)

        htbuck.insert(vector4, fitness4)
        htbuck.insert(vector5, fitness5)

        print len(htbuck.get_buckets())
        print htbuck.get_buckets()[0]
        print htbuck.get_buckets()[1]
        print htbuck.get_buckets()[2]
        print htbuck.get_buckets()[3]

        print '-1: ============================'
        print htbuck.get(vector3)
        print '6: ============================'
        print htbuck.get(vector3, 6)
        print '4: ============================'
        print htbuck.get(vector3, 4)
        print '2: ============================'
        print htbuck.get(vector3, 2)
