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
                  'landmark_buggy': ['reg_cost', 'bias', 'reg_cost_sq', 'reg_cost', 'bias_sq', 'bias', 'sc', 'rope_dist', 'landmark']}

FEATURE_LENGTHS = {'reg_cost_sq': 1,
                   'reg_cost': 1,
                   'bias_sq': 148,
                   'bias': 148,
                   'sc': 32,
                   'rope_dist': 3,
                   'landmark': 70}

# Set to [] if you want all landmark indices to be selected.
# Otherwise, only the indices indicated in LANDMARK_INDICES are selected.
LANDMARK_INDICES = []
#LANDMARK_INDICES = [ 0,  1,  2,  3,  4,  5, 10, 12, 15, 17, 19, 20, 23, 24, 25, 28, 29,
                    #30, 32, 33, 34, 36, 37, 38, 42, 44, 50, 53, 56, 59, 61, 62, 63, 64,
                    #65, 66, 67]

def subselect_feature(args, input_feature):
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
    return output_feature

def subselect_features(args):
    input_constraints = h5py.File(args.constraintfile, 'r')
    output_constraints = h5py.File(args.output_constraintfile, 'w')
    counter = 0
    for i in range(len(input_constraints)):
        key = str(i)
        counter += 1
        if counter%10000 == 0:
            print "# Constraints Finished: ", counter
        new_group = output_constraints.create_group(key)
        for group_key in input_constraints[key].keys():
            if group_key == 'exp_features' or group_key == "rhs_phi":
                new_group[group_key] = subselect_feature(args, input_constraints[key][group_key][()])
            else:
                new_group[group_key] = input_constraints[key][group_key][()]
    input_constraints.close()
    output_constraints.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('constraintfile')
    parser.add_argument('output_constraintfile')
    parser.add_argument('input_features', choices=['bias', 'quad', 'sc', 'rope_dist', 'landmark', 'landmark_buggy'])
    parser.add_argument('output_features', choices=['bias', 'quad', 'sc', 'rope_dist', 'landmark'])
    parser.add_argument('--rbf', action='store_true')
    args = parser.parse_args()

    # Commenting out this assert because we may need to pass in "landmark" for both input and output features, for subselecting
    # from the landmark features.
    #assert args.input_features != args.output_features, "Input and output features are the same, so there is nothing to be done"

    # Make sure the output features are compatible with the input
    if args.input_features == 'bias':
        assert args.output_features in ['bias'], "For bias input features, output must be bias"
    if args.input_features == 'quad':
        assert args.output_features in ['bias', 'quad'], "For quad input features, output must be either bias or quad"
    if args.input_features == 'sc':
        assert args.output_features in ['bias', 'sc'], "For sc input features, output must be either bias or sc"
    if args.input_features == 'rope_dist':
        assert args.output_features in ['bias', 'sc', 'rope_dist'], "For rope_dist input features, output must be either bias, sc, or rope_dist"
    if args.input_features == 'landmark':
        assert args.output_features in ['bias', 'quad', 'sc', 'rope_dist', 'landmark'], "For landmark input features, output must be either bias, quad, sc, or rope_dist"

    subselect_features(args)
    #import profile
    #profile.run("subselect_features(args)")
