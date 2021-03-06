#!/usr/bin/env python
#
# Run this script with ./subselect_constraints <constraint_file> <output_constraint_file> <input_features> <output_features>
#
# Assumes that the ids in the input constraint_file are consecutive integers,
# starting from 0.
# Options for input features are {bias, quad, sc, rope_dist, landmark} and
# options for output features are the types of features included in the type
# of the input feature.

# Input Feature | Allowed Output Features
# bias          | bias
# quad          | bias, quad
# sc            | bias, sc
# rope_dist     | bias, sc, rope_dist
# landmark      | bias, quad, sc, rope_dist, landmark
#
# Note: Make sure the FEATURE_ORDERS and FEATURE_LENGTHS vectors are correct
# for your needs!

import argparse
import h5py
import numpy as np

FEATURE_ORDERS = {'bias': ['reg_cost', 'bias'],
                  'quad': ['reg_cost_sq', 'reg_cost', 'bias_sq', 'bias'],
                  'sc': ['reg_cost', 'bias', 'sc'],
                  'rope_dist': ['reg_cost', 'bias', 'sc', 'rope_dist'],
                  'landmark': ['reg_cost_sq', 'reg_cost', 'bias_sq', 'bias', 'sc', 'rope_dist', 'landmark'],
                  'landmark_buggy': ['reg_cost', 'bias', 'reg_cost_sq', 'reg_cost', 'bias_sq', 'bias', 'sc', 'rope_dist', 'landmark'],
                  'quad_landmark': ['reg_cost_sq', 'reg_cost', 'bias_sq', 'bias', 'landmark'],
                  'quad_landmark_noregcostsq': ['reg_cost', 'bias_sq', 'bias', 'landmark'],
                  'ensemble': ['reg_cost_sq', 'reg_cost', 'bias_sq', 'bias', 'sc', 'rope_dist', 'landmark', 'done_bias', 'done_regcost', 'is_knot'],
                  'ensemble_nogoal': ['reg_cost_sq', 'reg_cost', 'bias_sq', 'bias', 'sc', 'rope_dist', 'landmark'],
                  'traj_diff': ['reg_cost_sq', 'reg_cost', 'bias_sq', 'bias', 'sc', 'rope_dist', 'landmark', 'done_bias', 'done_regcost', 'is_knot', 'traj_diff'],
                  'traj_diff_nogoal': ['reg_cost_sq', 'reg_cost', 'bias_sq', 'bias', 'sc', 'rope_dist', 'landmark', 'traj_diff']}

FEATURE_LENGTHS = {'reg_cost_sq': 1,
                   'reg_cost': 1,
                   'bias_sq': 148,
                   'bias': 148,
                   'sc': 32,
                   'rope_dist': 3,
                   'landmark': 70,
                   'done_bias': 1,
                   'done_regcost': 1,
                   'is_knot': 1,
                   'traj_diff': 1}

# Set to [] if you want all landmark indices to be selected.
# Otherwise, only the indices indicated in LANDMARK_INDICES are selected.
LANDMARK_INDICES = []
#LANDMARK_INDICES = [ 0,  1,  2,  3,  4,  5, 10, 12, 15, 17, 19, 20, 23, 24, 25, 28, 29,
                    #30, 32, 33, 34, 36, 37, 38, 42, 44, 50, 53, 56, 59, 61, 62, 63, 64,
                    #65, 66, 67]

def subselect_feature(args, input_feature, len_input, len_output):
    assert len(input_feature) == len_input, "Length of input feature does not match input feature type"
    start_index = 0
    output_start_indices = [0]*len(FEATURE_ORDERS[args.output_features])
    for feature_type in FEATURE_ORDERS[args.input_features]:
        if feature_type in FEATURE_ORDERS[args.output_features]:
            output_start_indices[FEATURE_ORDERS[args.output_features].index(feature_type)] = start_index
        start_index += FEATURE_LENGTHS[feature_type]

    output_feature = []
    for idx, start_i in enumerate(output_start_indices):
        feature_length = FEATURE_LENGTHS[FEATURE_ORDERS[args.output_features][idx]]
        if FEATURE_ORDERS[args.output_features][idx] == 'landmark':
            all_landmark = input_feature[start_i:start_i + feature_length]
            if args.rbf:
                all_landmark = np.exp(-np.square(all_landmark))
                all_landmark /= np.linalg.norm(all_landmark, 1)
            if LANDMARK_INDICES:
                output_feature.extend([all_landmark[i] for i in LANDMARK_INDICES])
            else:
                output_feature.extend(all_landmark)
        else:
            output_feature.extend(input_feature[start_i:start_i + feature_length])
    assert len(output_feature) == len_output, "Length of output feature does not match output feature type"
    return output_feature

def subselect_features(args):
    input_constraints = h5py.File(args.constraintfile, 'r')
    output_constraints = h5py.File(args.output_constraintfile, 'w')
    counter = 0

    len_input = 0
    len_output = 0
    for f in FEATURE_ORDERS[args.input_features]:
        len_input += FEATURE_LENGTHS[f]
    for f in FEATURE_ORDERS[args.output_features]:
        len_output += FEATURE_LENGTHS[f]

    for i in range(len(input_constraints)):
        key = str(i)
        counter += 1
        if counter%10000 == 0:
            print "# Constraints Finished: ", counter
        new_group = output_constraints.create_group(key)
        for group_key in input_constraints[key].keys():
            if group_key == 'exp_features' or group_key == "rhs_phi":
                new_group[group_key] = subselect_feature(args, input_constraints[key][group_key][()], len_input, len_output)
            else:
                new_group[group_key] = input_constraints[key][group_key][()]
    input_constraints.close()
    output_constraints.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('constraintfile')
    parser.add_argument('output_constraintfile')
    parser.add_argument('input_features', choices=['bias', 'quad', 'sc', 'rope_dist', 'landmark', 'landmark_buggy', 'quad_landmark', 'quad_landmark_noregcostsq', 'ensemble', 'ensemble_nogoal', 'traj_diff', 'traj_diff_nogoal'])
    parser.add_argument('output_features', choices=['bias', 'quad', 'sc', 'rope_dist', 'landmark', 'quad_landmark', 'quad_landmark_noregcostsq', 'ensemble', 'ensemble_nogoal', 'traj_diff', 'traj_diff_nogoal'])
    parser.add_argument('--rbf', action='store_true')
    args = parser.parse_args()

    # Commenting out this assert because we may need to pass in "landmark" for both input and output features, for subselecting
    # from the landmark features.
    #assert args.input_features != args.output_features, "Input and output features are the same, so there is nothing to be done"

    subselect_features(args)
    #import profile
    #profile.run("subselect_features(args)")
