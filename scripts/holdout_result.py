#!/usr/bin/env python

import argparse
import h5py
import matplotlib.pyplot as plt
import pylab
import colorsys
import numpy as np
import IPython as ipy
import os

def save_rope_nodes(rope_nodes, fname):
    plt.clf()
    plt.cla()
    links_z = (rope_nodes[:-1,2] + rope_nodes[1:,2])/2.0
#     min_z = min(links_z)
#     max_z = max(links_z)
#     color_z = (links_z - min_z)/(max_z-min_z)
#     for i in range(rope_nodes.shape[0]-1):
#         plt.plot(rope_nodes[i:i+2,0], rope_nodes[i:i+2,1], c=(color_z[i],0,1.0-color_z[i]), linewidth=6)
    for i in np.argsort(links_z):
        f = float(i)/(links_z.shape[0]-1)
        plt.plot(rope_nodes[i:i+2,0], rope_nodes[i:i+2,1], c=colorsys.hsv_to_rgb(f,1,1), linewidth=6)
    plt.axis("equal")
    plt.savefig(fname)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", type=str)
    parser.add_argument("images_dir", type=str)
    parser.add_argument("--holdout_file", type=str)
    args = parser.parse_args()

    result_file = h5py.File(args.results_file, 'r')
    holdout_file = None
    if args.holdout_file:
        holdout_file = h5py.File(args.holdout_file, 'r')
        assert len(result_file) == len(holdout_file)
    
    for i_task, task_info in result_file.iteritems():
        if holdout_file:
            fname = os.path.join(args.images_dir, "task_%s" % (str(i_task)))
            save_rope_nodes(holdout_file[str(i_task)]['rope_nodes'][()], fname)
            print "saved ", fname
        for i_step, step_info in task_info.iteritems():
            try:
                rope_nodes = step_info['rope_nodes'][()]
            except:
                rope_nodes = step_info
            fname = os.path.join(args.images_dir, "task_%s_step_%s" % (str(i_task), str(i_step)))
            save_rope_nodes(rope_nodes, fname)
            print "saved ", fname
