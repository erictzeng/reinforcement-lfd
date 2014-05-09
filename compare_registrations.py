#!/usr/bin/env python

# Conducts experiments for comparing TPS-RPM and Reg4, with and without visual
# features.
#
# Arguments:
#     input_file: h5 file with the key "orig_cloud" corresponding to a numpy
#                 array containing [x,y,r,b,g] info for the original points, and
#                 all other keys corresponding to warped point clouds in the
#                 same format

import argparse, h5py, numpy as np
from rapprentice import registrations

# TODO: Add in visual features
def run_experiments(input_file):
    clouds = h5py.File(input_file)
    x_nd = clouds['orig_cloud'][()][:,:2]

    for k in clouds:
        if k == 'orig_cloud':
            continue
        print "TESTING FOR WARP", k
        y_md = clouds[k][()][:,:2]
        f = registrations.sim_annealing_registration(x_nd, y_md,
                registrations.rpm_em_step, plotting=1, plot_cb = registrations.plot_callback)
        print "sim_annealing_registration rpm_em_step warps"
        print "Warp of [1, 1]:", f.transform_points(np.asarray([[1,1]]))
        print "Warp of [1, 1.5]:", f.transform_points(np.asarray([[1,1.5]]))
        print "Warp of [1, 2]:", f.transform_points(np.asarray([[1,2]]))

        f = registrations.sim_annealing_registration(x_nd, y_md,
                registrations.reg4_em_step, plotting=1, plot_cb = registrations.plot_callback)
        print "sim_annealing_registration rpm_em_step warps"
        print "Warp of [1, 1]:", f.transform_points(np.asarray([[1,1]]))
        print "Warp of [1, 1.5]:", f.transform_points(np.asarray([[1,1.5]]))
        print "Warp of [1, 2]:", f.transform_points(np.asarray([[1,2]]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)

    args = parser.parse_args()
    run_experiments(args.input_file)

if __name__ == "__main__":
    main()
