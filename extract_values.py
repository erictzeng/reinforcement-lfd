#!/usr/bin/env python

import argparse
usage="""
Generate holdout set with
./extract_values.py <results_file>

Example
./extract_values.py "data/results/bellman_multi_quad_500_50/holdout_result.h5"
"""
parser = argparse.ArgumentParser(usage=usage)
parser.add_argument("results_file", type=str)

args = parser.parse_args()

import h5py
import matplotlib.pyplot as plt
import pylab
import numpy as np
import IPython as ipy
import os

if __name__ == '__main__':

    result_file = h5py.File(args.results_file, 'r')
    values = np.zeros((len(result_file),0))
    for i_task, task_info in result_file.iteritems():
        if len(task_info) > values.shape[1]:
            values = np.c_[values, np.zeros((len(result_file), len(task_info)-values.shape[1]))]
        for i_step, step_info in task_info.iteritems():
            values[int(i_task), int(i_step)] = np.max(step_info['values'][()])
    print values
    plt.plot(values.transpose())
    plt.show()
    ipy.embed()
