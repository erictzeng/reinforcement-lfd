#!/usr/bin/env python
import h5py
import argparse
import IPython as ipy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('infile')
    parser.add_argument('outfile')
    args = parser.parse_args()
    
    outfile = h5py.File(args.outfile, 'w-') # fail if file already exists
    infile = h5py.File(args.infile, 'r')
            
    constraint_ctr = 0

    constr_trajectories = {}
    for key_i in range(len(infile)):
        constr = infile[str(key_i)]
        slack_name = constr['xi'][()]
        if slack_name.startswith('yi'):
            traj_i = slack_name # there is only one slack variable yi per trajectory
            if traj_i not in constr_trajectories:
                constr_trajectories[slack_name] = []
            constr_trajectories[slack_name].append(constr)
        else:
            outfile.copy(constr, str(constraint_ctr))
            constraint_ctr += 1
            
    for constr_traj in constr_trajectories.values():
        if len(constr_traj) > 1:
            for constr in constr_traj:
                outfile.copy(constr, str(constraint_ctr))
                constraint_ctr += 1

    infile.close()
    outfile.close()
