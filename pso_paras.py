# -*- coding: utf-8 -*-
# !/usr/bin/env python

__author__ = 'jeff'

from traits.api import HasTraits, Str, Int, Date, Bool, Float

class PsoParas(HasTraits):
    c1 = Float
    c2 = Float
    dim = Int
    pop_number = Int
    max_gen = Int
    val_max = Float
    val_min = Float
    spd_max = Float
    break_condition = Float
    show_figure = Bool
    times = Int
