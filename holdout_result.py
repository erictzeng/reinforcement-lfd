#!/usr/bin/env python

import argparse
usage="""
Generate holdout set with
./holdout_result.py <results_file> <images_dir>

Example
./holdout_result.py "data/results/multi_quad_100/holdout_result.h5" "data/results/multi_quad_100/images"
"""
parser = argparse.ArgumentParser(usage=usage)
parser.add_argument("results_file", type=str)
parser.add_argument("images_dir", type=str)

args = parser.parse_args()

import h5py
import matplotlib.pyplot as plt
import pylab
import colorsys
import numpy as np
import IPython as ipy
import os

if __name__ == '__main__':

    result_file = h5py.File(args.results_file, 'r')

    for i_task, task_info in result_file.iteritems():
        for i_step, step_info in task_info.iteritems():
            try:
                rope_nodes = step_info['rope_nodes'][()]
            except:
                rope_nodes = step_info
            plt.clf()
            plt.cla()
            links_z = (rope_nodes[:-1,2] + rope_nodes[1:,2])/2.0
#             min_z = min(links_z)
#             max_z = max(links_z)
#             color_z = (links_z - min_z)/(max_z-min_z)
#             for i in range(rope_nodes.shape[0]-1):
#                 plt.plot(rope_nodes[i:i+2,0], rope_nodes[i:i+2,1], c=(color_z[i],0,1.0-color_z[i]), linewidth=6)
            for i in np.argsort(links_z):
                f = float(i)/(links_z.shape[0]-1)
                plt.plot(rope_nodes[i:i+2,0], rope_nodes[i:i+2,1], c=colorsys.hsv_to_rgb(f,1,1), linewidth=6)
            plt.axis("equal")
            fname = os.path.join(args.images_dir, "task_%s_step_%s" % (str(i_task), str(i_step)))
            plt.savefig(fname)
            print "saved ", fname
