#!/usr/bin/env python

import argparse
usage="""
Generate holdout set with
./quadraticize_constraints.py <input_constraints_file> <output_constraints_file>

Example
./quadraticize_constraints.py "data/mm_constraints_1.h5" "data/mm_constraints_1_quadratic.h5"

If the input constraint file has weights, those weights are ignored
"""
parser = argparse.ArgumentParser(usage=usage)
parser.add_argument("infile", type=str)
parser.add_argument("outfile", type=str)

args = parser.parse_args()

import h5py
import numpy as np
import IPython as ipy

def reformat_vector(v):
    n_affine = v.shape[0]-1
    v_new = np.zeros((2 + 2*n_affine,))
    s = v[0]
    v_new[0] = s**2
    v_new[1] = s
    v_new[2:2+n_affine] = s*v[1:]
    v_new[2+n_affine:] = v[1:]
    return v_new

if __name__ == '__main__':

    outfile = h5py.File(args.outfile, 'w-') # fail if file already exists
    infile = h5py.File(args.infile, 'r')

    for key, constr in infile.iteritems():
        if key == 'weights': continue
        
        exp_phi = constr['exp_features'][:]
        rhs_phi = constr['rhs_phi'][:]
        margin = float(constr['margin'][()])

        g = outfile.create_group(str(key))
        g['exp_features'] = reformat_vector(exp_phi)
        g['rhs_phi'] = reformat_vector(rhs_phi)
        g['margin'] = margin

    outfile.close()
    infile.close()
