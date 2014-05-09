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
from rapprentice import registration, registrations

def run_experiments(input_file):
    clouds = h5py.File(input_file)
    _,d = clouds['orig_cloud'][()].shape
    d = d - 3  # ignore the RGB values
    x_xyzrgb = clouds['orig_cloud'][()]
    x_nd = clouds['orig_cloud'][()][:,:d]

    for k in clouds:
        if k == 'orig_cloud':
            continue
        print "TESTING FOR WARP", k
        y_xyzrgb = clouds[k][()]
        y_md = clouds[k][()][:,:d]
        vis_costs_xy = registrations.ab_cost(x_xyzrgb, y_xyzrgb)


        print "Reg4 EM, w/ visual features"
        f = registrations.sim_annealing_registration(x_nd, y_md,
                registrations.reg4_em_step, vis_cost_xy = vis_costs_xy,
                plotting=1, plot_cb = registrations.plot_callback)

        print "Reg4 EM, w/o visual features"
        f = registrations.sim_annealing_registration(x_nd, y_md,
                registrations.reg4_em_step,
                plotting=1, plot_cb = registrations.plot_callback)

        print "RPM EM, w/ visual features"
        f = registrations.sim_annealing_registration(x_nd, y_md,
                registrations.rpm_em_step, vis_cost_xy = vis_costs_xy,
                plotting=1, plot_cb = registrations.plot_callback)

        print "RPM EM, w/o visual features"
        f = registrations.sim_annealing_registration(x_nd, y_md,
                registrations.rpm_em_step,
                plotting=1, plot_cb = registrations.plot_callback)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)

    args = parser.parse_args()
    run_experiments(args.input_file)

if __name__ == "__main__":
    main()
