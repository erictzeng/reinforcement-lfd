#!/usr/bin/env python
import argparse, h5py, random, os.path
import numpy as np
from do_task_label import check_outfile, write_flush
import IPython as ipy

CONSTR_KEYS = ['example', 'exp_features', 'margin', 'rhs_phi', 'xi']

def extract_trajs(exfile):
    trajs = []
    cur_traj = None
    for k in range(len(exfile)):
        key = str(k)        
        if exfile[key]['pred'][()] == key:
            if cur_traj: trajs.append(cur_traj)
            cur_traj = []
        cur_traj.append(key)
    if cur_traj: trajs.append(cur_traj)
    return trajs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('constraintfile')
    parser.add_argument('examplefile')
    parser.add_argument('--subsample_size', type=int)
    parser.add_argument('outfile')
    args = parser.parse_args()

    cfile = h5py.File(args.constraintfile, 'r')
    exfile = h5py.File(args.examplefile)
    outfile = h5py.File(args.outfile, 'w')
    check_outfile(exfile)

    trajs = extract_trajs(exfile)
    random.shuffle(trajs)

    num_copied = 0
    copy_keys = []
    while num_copied < args.subsample_size:
        t = trajs.pop()
        num_copied += len(t) - 1 # ignore the endstate
        copy_keys.extend(t)
    
    for k in range(len(cfile)):
        key = str(k)
        g = cfile[key]
        constraint_id = g['example'][()]
        try:
            mentioned_keys = [constraint_id]
        except ValueError:
            mentioned_keys = [x for x in constraint_id.split('-')[:2]]
        if any(x in copy_keys for x in mentioned_keys):
            g_items = zip([(k, g[k][()]) for k in g])
            write_flush(outfile, g_items)
    ipy.embed()
    cfile.close()
    exfile.close()
    outfile.close()
    
