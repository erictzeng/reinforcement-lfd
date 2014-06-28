#!/usr/bin/env python

import argparse
usage="""
Generate holdout set with
./extract_values.py <resultfile>

Example
./extract_values.py "data/results/bellman_multi_quad_500_50/holdout_result.h5"
"""
parser = argparse.ArgumentParser(usage=usage)
parser.add_argument("resultfile", type=str)
parser.add_argument("--outfile", type=str)
parser.add_argument("--remove_outlier", action="store_true")
parser.add_argument("--font_size", type=int)

args = parser.parse_args()

import h5py
import matplotlib
import matplotlib.pyplot as plt
import pylab
import numpy as np
import IPython as ipy
import os

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True

if __name__ == '__main__':

    result_file = h5py.File(args.resultfile, 'r')
    values = np.zeros((len(result_file),0))
    for i_task, task_info in result_file.iteritems():
        if len(task_info) > values.shape[1]:
            values = np.c_[values, np.zeros((len(result_file), len(task_info)-values.shape[1]))]
        for i_step, step_info in task_info.iteritems():
            values[int(i_task), int(i_step)] = np.max(step_info['values'][()])
    if args.remove_outlier:
        inds = [i for (i,v) in enumerate(values[:,2]) if v==0]
        print "removing indices from values", inds
        values = np.delete(values, inds, axis=0)
    print values

    plt.xlabel('step')
    plt.ylabel('state value')
    plt.xticks(np.arange(0,6))

    if args.font_size:
        ax = plt.gca()
        
        txt = ax.get_xlabel()
        txt_obj = ax.set_xlabel(txt)
        txt_obj.set_fontsize(args.font_size)

        txt = ax.get_ylabel()
        txt_obj = ax.set_ylabel(txt)
        txt_obj.set_fontsize(args.font_size)
    
        ticks = ax.get_xticklabels() + ax.get_yticklabels()
        for t in ticks:
            t.set_fontsize(args.font_size)
    
    plt.plot(range(1,6), values.transpose())
    
    if args.outfile:
        plt.savefig(args.outfile)
    else:
        plt.show()
        ipy.embed()
