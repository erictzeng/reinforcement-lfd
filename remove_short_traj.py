#!/usr/bin/env python
#
# Run this script with ./remove_short_traj.py <labeled_examples_file> <output_examples_file>
#
# Outputs a new labeled examples file, excluding trajectories with length
# less than 4 (including endstate). Renumbers the trajectories so that their
# ids are in consecutive numerical order, starting from 0.
# Assumes ids of the labelled examples are in consecutive numerical order,
# starting from 0.

import argparse, h5py

def remove_short_traj(examples, output):
    num_examples = len(examples.keys())
    output_id = 0  # To keep track of renumbering
    prev_start = 0
    for i in range(num_examples):
        k = str(i)
        pred = int(examples[k]['pred'][()])
        if pred == i and i != 0:
            if i - prev_start < 4:  # trajectory has less than 4 (inc. endstate)
                print "Removing trajectory starting at id ", prev_start, ", length: ", i - prev_start
                for i_rm in range(i - prev_start):
                    output_id -= 1
                    print "Deleting output id ", output_id
                    del output[str(output_id)]
                print "Adding again at output id ", output_id
            prev_start = i

        new_group = output.create_group(str(output_id))
        for group_key in examples[k].keys():
            # Update the value of 'pred' correctly (with the renumbering)
            if group_key == 'pred':
                assert pred == i or pred == i-1, "Invalid predecessor value for %i"%i
                if pred == i:
                    new_group[group_key] = str(output_id)
                else:
                    new_group[group_key] = str(output_id - 1)
            else:
                new_group[group_key] = examples[k][group_key][()]
        output_id += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('examples_file')
    parser.add_argument('output_examples_file')
    args = parser.parse_args()

    examples = h5py.File(args.examples_file, 'r')
    output = h5py.File(args.output_examples_file, 'w')
    remove_short_traj(examples, output)
    examples.close()
    output.close()
