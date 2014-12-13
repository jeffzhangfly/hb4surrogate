# -*- coding: utf-8 -*-
# !/usr/bin/env python

__author__ = 'jeff'
import pso
import math
import time
import numpy as np
from traits.etsconfig.api import ETSConfig
import matplotlib.pyplot as plt

ETSConfig.toolkit = 'qt4'  # 'wx'
from traitsui.api import View, Group, Item, Handler
import pso_paras as pparas


def Sphere(vars):
    s = 0
    dim = vars.shape[0]
    for i in range(dim):
        s += vars[i] * vars[i]

    return s


def Rastrigin(vars):
    s = 0
    dim = vars.shape[0]
    for i in range(dim):
        s += 100 * (vars[i] * vars[i] - math.cos(2 * math.pi * vars[i]) + 1)

    return s


def Griewank(vars):
    tmp1 = 0
    tmp2 = 1
    dim = vars.shape[0]
    for i in range(1, dim + 1):
        tmp1 += vars[i - 1] * vars[i - 1] / float(4000)
        tmp2 *= math.cos(vars[i - 1] / math.sqrt(i))

    return tmp1 + tmp2 + 1


def Rosenbrock(vars):
    s = 0
    dim = vars.shape[0]
    for i in range(dim - 1):
        s += (1 - vars[i]) * (1 - vars[i]) + \
             100 * (vars[i + 1] - vars[i] * vars[i]) * (vars[i + 1] - vars[i] * vars[i])
    return s


def main():
    comp_view = View(
        Group(
            Group(
                Item('c1', label=u'c1 of PSO'),
                Item('c2', label=u'c2 of PSO'),
                Item('dim', label=u'维数'),
                Item('pop_number', label=u'种群规模'),
                Item('max_gen', label=u'最大迭代代数'),
                Item('val_max', label=u'个体位置最大值'),
                Item('val_min', label=u'个体位置最小值'),
                Item('break_condition', label=u'适应值终止阈值'),
                Item('show_figure', label=u'显示进化曲线'),
                Item('times', label=u'测试次数'),
                label=u'优化参数设定',
                show_border=True
            ),
            orientation='horizontal'
        ),
        title=u'PSO',
        kind='live',
        buttons=['OK']
    )

    paras = pparas.PsoParas()
    paras.c1 = 0.9
    paras.c2 = 1.2
    paras.dim = 10
    paras.pop_number = 200
    paras.max_gen = 200
    paras.val_max = 5.12
    paras.val_min = -5.12
    paras.break_condition = 0
    paras.show_figure = False

    paras.times = 1

    paras.configure_traits(view=comp_view)  # , handle=PlotHandle())

    print "##############################################"
    print "Calc..."
    start = time.time()
    use_hash = True
    hash_precise = [6, 5, 4, 3]
    min_neighbour_number = [5, 10, 20, 30]
    # hash_precise = [3, 2, 4]
    # hash_precise = [6, 5, 4, 3 ]

    ##################################################################
    # statistic
    total_totalTime = 0
    total_fitnessEvalTimes = 0
    total_directHitTimes = 0
    total_approximateTimes = 0
    total_approximateTimesPerBucket = [0 for i in range(len(hash_precise))]
    total_approximated_better_than_gbest_then_calc = 0
    total_timeInsert = 0
    total_timeGet = 0
    total_timeFitnessEval = 0

    total_totalTime_values = []
    total_fitnessEvalTimes_values = []
    total_directHitTimes_values = []
    total_approximateTimes_values = []
    total_approximateTimesPerBucket_values = [[] for i in range(len(hash_precise))]
    total_approximated_better_than_gbest_then_calc_values = []
    total_timeInsert_values = []
    total_timeGet_values = []
    total_timeFitnessEval_values = []

    gbest_sum = 0
    gbests = []

    for i in range(paras.times):
        print "No." + str(i)
        p = pso.Pso(paras.c1, paras.c2, paras.dim, paras.pop_number, paras.max_gen,
                    paras.val_max, paras.val_min, Rastrigin, paras.break_condition, use_hash, hash_precise,
                    min_neighbour_number)
        gbest, gbest_particle, gbest_history, pbest_avg_history = p.iterate()
        end = time.time()
        totalTime = end - start

        gbests.append(gbest)
        gbest_sum += gbest

        fitnessEvalTimes, directHitTimes, approximateTimes, precisions, approximateTimesPerBucket, \
        approximated_better_than_gbest_then_calc, timeInsert, timeGet, timeFitnessEval = p.statistics()

        total_totalTime += totalTime
        total_totalTime_values.append(totalTime)

        total_fitnessEvalTimes += fitnessEvalTimes
        total_fitnessEvalTimes_values.append(fitnessEvalTimes)

        total_directHitTimes += directHitTimes
        total_directHitTimes_values.append(directHitTimes)

        total_approximateTimes += approximateTimes
        total_approximateTimes_values.append(approximateTimes)

        for i in range(len(hash_precise)):
            total_approximateTimesPerBucket[i] += approximateTimesPerBucket[i]
            total_approximateTimesPerBucket_values[i].append(approximateTimesPerBucket[i])

        total_approximated_better_than_gbest_then_calc += approximated_better_than_gbest_then_calc
        total_approximated_better_than_gbest_then_calc_values.append(approximated_better_than_gbest_then_calc)

        total_timeInsert += timeInsert
        total_timeInsert_values.append(timeInsert)

        total_timeGet += timeGet
        total_timeGet_values.append(timeGet)

        total_timeFitnessEval += timeFitnessEval
        total_timeFitnessEval_values.append(timeFitnessEval)

    gbest_avg = gbest_sum / paras.times

    print "##############################################"
    print "Done! RESULT:"
    print
    print "Total Elapsed: " + str(total_totalTime) + "(s)" \
          + " | Average Elapsed: " + str(total_totalTime / paras.times) + "(s)" \
          + " | std: " + str(std(total_totalTime_values)) + "(s)"

    if use_hash:
        print "Total Insert Hash Elapsed: " + str(total_timeInsert) + "(s)"\
              + " | Average Elapsed: " + str(total_timeInsert / paras.times) + "(s)" \
              + " | std: " + str(std(total_timeInsert_values))

        print "Total Get Hash Elapsed: " + str(total_timeGet) + "(s)" \
              + " | Average Elapsed: " + str(total_timeGet / paras.times) + "(s)"\
              + " | std: " + str(std(total_timeGet_values))

        print "Total Fitness Evaluate Elapsed: " + str(total_timeFitnessEval) + "(s)"\
              + " | Average Elapsed: " + str(total_timeFitnessEval / paras.times) + "(s)" \
              + " | std: " + str(std(total_timeFitnessEval_values))
    print
    print "Total Eval Times: " + str(paras.pop_number * paras.max_gen * paras.times)
    print "  -> Total Fitness Eval Times: " + str(total_fitnessEvalTimes) \
              + " | Average Times: " + str((0.0 + total_fitnessEvalTimes) / paras.times) \
              + " | std: " + str(std(total_fitnessEvalTimes_values))

    if use_hash:
        if paras.times == 1:
            print " Hash tabel length:"
            for i in range(len(hash_precise) + 1):
                print "  ->" + str(p.get_hash_buckets_precise(i)) + ": " + str(p.get_hash_buckets_len(i) - 1)
            print

        print "  -> Total Direct Hit Times: " + str(total_directHitTimes)  \
              + " | Average Times: " + str((0.0 + total_directHitTimes) / paras.times) \
              + " | std: " + str(std(total_directHitTimes_values))

        print "  -> Total Approximate Times: " + str(total_approximateTimes)  \
              + " | Average Times: " + str((0.0 + total_approximateTimes) / paras.times) \
              + " | std: " + str(std(total_approximateTimes_values))

        for i in range(len(precisions)):
            print "    -> Bucket " + str(precisions[i]) + ": " + str(total_approximateTimesPerBucket[i])  + \
                  " | Average Times: " + str((0.0 + total_approximateTimesPerBucket[i]) / paras.times) +\
                  " | std: " + str(std(total_approximateTimesPerBucket_values[i]))

        print "Total Approximated value better than gbest: " + str(total_approximated_better_than_gbest_then_calc) + " times!"  \
              + " | Average Times: " + str((0.0 + total_approximated_better_than_gbest_then_calc) / paras.times) + " times!" \
              + " | std: " + str(std(total_approximated_better_than_gbest_then_calc_values))

    print
    print "Result Average: "
    for gb in gbests:
        print gb
    print "gbest Average: " + str(gbest_avg)
    # print gbest_particle
    print "##############################################"

    if paras.show_figure:
        # figure
        x = np.linspace(0, paras.max_gen - 1, paras.max_gen)
        plt.figure(figsize=(5, 4))

        plt.plot(x, gbest_history, label="gbest", color="red", linewidth=2)
        plt.plot(x, pbest_avg_history, "b--", label="pbest mean", linewidth=2)

        plt.xlabel("Generation", fontsize=18)
        plt.ylabel("Fitness", fontsize=18)
        # plt.title("Evolutionary Curve", fontsize=18, color="blue")

        xaxis = plt.gca().xaxis
        yaxis = plt.gca().yaxis
        for label in xaxis.get_ticklabels():
            # label.set_color("red")
            # label.set_rotation(45)
            label.set_fontsize(14)

        for label in yaxis.get_ticklabels():
            label.set_fontsize(14)

        plt.subplots_adjust(top=0.97, bottom=0.15, left=0.20, right=0.96)
        plt.gca().xaxis.grid(True, which="major")
        plt.gca().yaxis.grid(True, which="major")

        # plt.ylim(-1.2, 1.2)
        plt.legend()
        plt.show()

def std(values):
    if len(values) == 1:
        return 0

    sum = 0
    sum_sub = 0
    for v in values:
        sum += v
    avg = (0.0 + sum) / (0.0 + len(values))
    for v in values:
        sum_sub += (v - avg) * (v - avg)
    std = math.sqrt(sum_sub / (0.0 + len(values) - 1))
    return std

if __name__ == "__main__":
    main()