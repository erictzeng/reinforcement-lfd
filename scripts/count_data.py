#!/usr/bin/env python

import h5py

f = h5py.File('expert_demos.h5', 'r')

num_keys = 0

for g in f.values():
    if g['action'][()].startswith('endstate'):
        continue
    num_keys += 1

print "there are {} data points".format(num_keys)
