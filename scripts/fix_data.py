#!/usr/bin/env python
import h5py
import argparse
from do_task_label import check_outfile

parser = argparse.ArgumentParser()
parser.add_argument('h5file')
args = parser.parse_args()

f = h5py.File(args.h5file, 'r+')
for k in f:
    if f[k]['pred'][()] == '0':
        del f[k]['pred']
        f[k]['pred'] = str(k)

check_outfile(f)
