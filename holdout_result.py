#!/usr/bin/env python

import argparse
usage="""
Generate holdout set with
./holdout_result.py <results_file> <images_dir>

Example
./holdout_result.py "data/holdout_result.h5" "eval"
"""
parser = argparse.ArgumentParser(usage=usage)
parser.add_argument("results_file", type=str)
parser.add_argument("images_dir", type=str)

args = parser.parse_args()

import h5py
import matplotlib
import pylab
import IPython as ipy
import os

if __name__ == '__main__':

    result_file = h5py.File(args.results_file, 'r')

    for i_task, i_step_rope_nodes in result_file.iteritems():
        for i_step, rope_nodes in i_step_rope_nodes.iteritems():
            matplotlib.pyplot.clf()
            matplotlib.pyplot.cla()
            matplotlib.pyplot.scatter(rope_nodes[:,0], rope_nodes[:,1])
            matplotlib.pyplot.axis("equal")
            fname = os.path.join(args.images_dir, "task_%s_step_%s" % (str(i_task), str(i_step)))
            matplotlib.pyplot.savefig(fname)
            print "saved ", fname
