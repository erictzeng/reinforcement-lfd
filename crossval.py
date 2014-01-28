from rope_qlearn import optimize_model
from util import suppress_stdout

import argparse
from itertools import product
from multiprocessing import Pool
import os
import re
import sys

# slacks - one global weight
# F

#SLACK_COEFF_VALS = [10 ** i for i in range(-1, 4)]
#BELLMAN_COEFF_VALS = [10 ** i for i in range(-1, 4)]

SLACK_COEFF_VALS = [1]
BELLMAN_COEFF_VALS = [1, 10]

def parse_modelname(modelfile):
    modelfile = os.path.basename(modelfile)
    match = re.match(r'(.*)_model.mps$', modelfile)
    if match:
        return match.group(1)
    else:
        print "Couldn't parse modelname {}.".format(modelfile)
        return raw_input('Please type a modelname: ')

def get_weightname(prefix, slack_coeff, bellman_coeff):
    info = {'prefix': prefix, 'slack_coeff': slack_coeff, 'bellman_coeff': bellman_coeff}
    return '{prefix}_slack{slack_coeff:g}_bellman{bellman_coeff:g}_weights.h5'.format(**info)

def fake_args(modelfile, weightfile, slack_coeff, bellman_coeff):
    args = argparse.Namespace()
    args.modelfile = modelfile
    args.weightfile = weightfile
    args.actionfile = 'data/misc/actions.h5'
    args.demofile = 'data/misc/all_labels.h5'
    args.model = 'bellman'
    args.ensemble = True
    args.landmark_features = 'data/misc/landmarks/landmarks_70.h5'
    args.only_landmark = False
    args.rbf = True
    args.quad_features = False
    args.sc_features = False
    args.rope_dist_features = False
    args.C = slack_coeff
    args.D = 4.0*slack_coeff
    args.F = bellman_coeff
    args.save_memory = False
    args.gripper_weighting = False
    args.goal_constraints = False
    args.parallel = False
    return args

def optimize_models(modelfile, weightdir, slack_coeff_vals, bellman_coeff_vals, prefix=None):
    if prefix is None:
        prefix = parse_modelname(modelfile)
    args_list = []
    print '0% complete.',
    sys.stdout.flush()
    num_weights = len(slack_coeff_vals) * len(bellman_coeff_vals)
    for i, (slack_coeff, bellman_coeff) in enumerate(product(slack_coeff_vals, bellman_coeff_vals), 1):
        weightfile = os.path.join(weightdir, get_weightname(prefix, slack_coeff, bellman_coeff))
        with suppress_stdout():
            optimize_model(fake_args(modelfile, weightfile, slack_coeff, bellman_coeff))
        sys.stdout.write('\r{0:%} complete.'.format(float(i) / num_weights))
    print

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('modelfile')
    parser.add_argument('weightdir')
    args = parser.parse_args()

    print 'Slack coefficients:\t', SLACK_COEFF_VALS
    print 'Bellman coefficients:\t', BELLMAN_COEFF_VALS
    try:
        raw_input('Enter to continue, C-c to quit. ')
    except KeyboardInterrupt:
        print
        exit(1)

    if not os.path.exists(args.weightdir):
        os.makedirs(args.weightdir)

    optimize_models(args.modelfile, args.weightdir, SLACK_COEFF_VALS, BELLMAN_COEFF_VALS)
    print 'Done.'
