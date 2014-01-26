#!/usr/bin/env python

import argparse
import h5py

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('h5file', type=str)
    args = parser.parse_args()
    
    f = h5py.File(args.h5file, 'r')
    print "File %s has %i keys"%(args.h5file, len(f))
    f.close()
