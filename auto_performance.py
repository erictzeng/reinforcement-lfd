#!/usr/bin/env python

import argparse
import h5py
from knot_classifier import isKnot

def estimate_performance(results_file):
    if type(results_file) is str:
        results_file = h5py.File(results_file, 'r')

    num_knots = 0
    knot_inds = []
    not_inds = []
    for (i_task, task_info) in sorted(results_file.iteritems(), key=lambda item: int(item[0])):
        is_knot = False
        for i_step in range(len(task_info) - (1 if 'init' in task_info else 0)):
            step_info = task_info[str(i_step)]
            try:
                rope_nodes = step_info['rope_nodes'][()]
            except:
                rope_nodes = step_info
            
            if isKnot(rope_nodes):
                num_knots += 1
                is_knot = True
                break
        if is_knot:
            knot_inds.append(int(i_task))
        else:
            not_inds.append(int(i_task))
    
    return num_knots, knot_inds, not_inds

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", type=str)
    args = parser.parse_args()

    results_file = h5py.File(args.results_file, 'r')
    
    num_successes, _, not_inds = estimate_performance(args.results_file)
    print "not_inds", not_inds    
    print "Successes / Total: %d/%d" % (num_successes, len(results_file))
    print "Success rate:", float(num_successes)/float(len(results_file))
