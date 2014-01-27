#!/usr/bin/env python

import argparse
import h5py
from knot_classifier import isKnot

def estimate_performance(results_file):
    if type(results_file) is str:
        results_file = h5py.File(results_file, 'r')

    num_knots = 0
    for i_task in range(len(results_file)):
        task_info = results_file[str(i_task)]
        for i_step in range(len(task_info)):
            step_info = task_info[str(i_step)]
            try:
                rope_nodes = step_info['rope_nodes'][()]
            except:
                rope_nodes = step_info
            
            if isKnot(rope_nodes):
                num_knots += 1
                break
    
    return float(num_knots)/len(results_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", type=str)
    args = parser.parse_args()
    
    print "success rate is", estimate_performance(args.results_file)
