#!/usr/bin/env python

try:
    import gurobipy as grb
    USE_GUROBI = True
    GRB = grb.GRB # constants for gurobi
except ImportError:
    USE_GUROBI = False
import IPython as ipy
import numpy as np
import h5py, random, math, util, time
from numbers import Number
from pdb import pm, set_trace
import sys
import argparse
from joblib import Parallel, delayed
from rope_qlearn import registration_cost_cheap
from sklearn.svm import LinearSVC
import random
eps = 10**-8
MAX_ITER=1000
NUM_NEGATIVES=300

"""
stuff for training an SVM to recognize knots
"""

def exp_rc(cl1, cl2):
    return np.exp(-1*registration_cost_cheap(cl1, cl2)**2)

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('example_file', type=str)
    parser.add_argument('landmark_file', type=str)
    parser.add_argument('outfile', type=str)
    args = parser.parse_args()
    ex_file = h5py.File(args.example_file, 'r')
    landmark_file = h5py.File(args.landmark_file, 'r')
    supports = [g['cloud_xyz'][:] for g in landmark_file.itervalues()]
    num_features = len(landmark_file)
    # get indices of knots from examples
    # knots= []
    # others = []
    # for k,g in ex_file.iteritems():
    #     if g['knot'][()]:
    #         knots.append(g['cloud_xyz'][:])
    #     else:
    #         others.append(g['cloud_xyz'][:])
    # compute features
    # random.shuffle(others)
    # others = others[:NUM_NEGATIVES]
    # num_features = len(knots) + NUM_NEGATIVES
    features = np.zeros((len(ex_file), num_features))
    labels = np.zeros(len(ex_file))
    # supports = knots
    # supports.extend(others)
    outfile = h5py.File(args.outfile, 'w')
    compute_time = 0
    num_examples = len(ex_file)
    for i, ex_k in enumerate(ex_file):
        start = time.time()
        costs = Parallel(n_jobs=4,verbose=0)(delayed(registration_cost_cheap)(ex_file[ex_k]['cloud_xyz'][:], cl) 
                                                                for cl in supports)
        features[i,:] = costs
        labels[i] = ex_file[ex_k]['knot'][()]
        if 'features' in outfile:
            del outfile['features']
        if 'labels' in outfile:
            del outfile['labels']
        outfile['features'] = features
        outfile['labels'] = labels
        outfile.flush()
        compute_time = .5*(time.time() - start) + .5*compute_time
        eta = compute_time*(num_examples - i)
        sys.stdout.write('Computed features for {} examples, estimated time left:\t{}\r'.format(i, eta/float(60)))
        sys.stdout.flush()
    sys.stdout.write('\n')
    indices = range(len(labels))
    random.shuffle(indices)
    train = indices[:len(labels)/2]
    test = indices[len(labels)/2:]
    clf = LinearSVC(penalty='l1', dual=False)
    clf.fit(features[train], labels[train])
    errors = clf.predict(features[test]) - labels[test]
    print "errors", sum(abs(errors))
    ipy.embed()

        
