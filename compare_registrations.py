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

def run_experiments(input_file, plot_color):
    clouds = h5py.File(input_file)
    _,d = clouds['orig_cloud'][()].shape
    d = d - 3  # ignore the RGB values
    y_xyzrgb = clouds['orig_cloud'][()]
    y_md = clouds['orig_cloud'][()][:,:d]

    for k in clouds:
        if not k.startswith('warp'):
            continue
        print "TESTING:", k
        x_xyzrgb = clouds[k][()]
        x_nd = clouds[k][()][:,:d]
        vis_costs_xy = registrations.ab_cost(x_xyzrgb, y_xyzrgb)

        def plot_cb(x_nd, y_md, corr_nm, f):
            if plot_color:
                registrations.plot_callback(x_nd, y_md, corr_nm, f, x_color = x_xyzrgb[:,d:], y_color = y_xyzrgb[:,d:])
            else:
                registrations.plot_callback(x_nd, y_md, corr_nm, f)

        print "Reg4 EM, w/ visual features"
        f = registrations.sim_annealing_registration(x_nd, y_md,
                registrations.reg4_em_step, vis_cost_xy = vis_costs_xy,
                plotting=1, plot_cb = plot_cb)

        print "Reg4 EM, w/o visual features"
        f = registrations.sim_annealing_registration(x_nd, y_md,
                registrations.reg4_em_step,
                plotting=1, plot_cb = plot_cb)

        print "RPM EM, w/ visual features"
        f = registrations.sim_annealing_registration(x_nd, y_md,
                registrations.rpm_em_step, vis_cost_xy = vis_costs_xy,
                plotting=1, plot_cb = plot_cb)

        print "RPM EM, w/o visual features"
        f = registrations.sim_annealing_registration(x_nd, y_md,
                registrations.rpm_em_step,
                plotting=1, plot_cb = plot_cb)
        
        def plot_cb_bij(x_nd, y_md, xtarg_nd, corr_nm, wt_n, f):
            if plot_color:
                registrations.plot_callback(x_nd, y_md, corr_nm, f, res = (.3, .3, .12), x_color = x_xyzrgb[:,d:], y_color = y_xyzrgb[:,d:])
            else:
                registrations.plot_callback(x_nd, y_md, corr_nm, f, res = (.3, .3, .12))
        scaled_x_nd, _ = registration.unit_boxify(x_nd)
        scaled_y_md, _ = registration.unit_boxify(y_md)
        f,g = registration.tps_rpm_bij(scaled_x_nd, scaled_y_md, plot_cb=plot_cb_bij,
                                       plotting=1, rot_reg=np.r_[1e-4, 1e-4, 1e-1][:d], 
                                       n_iter=50, reg_init=10, reg_final=.1, outlierfrac=1e-2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("--plot_color", type=int, default=1)

    args = parser.parse_args()
    run_experiments(args.input_file, args.plot_color)

if __name__ == "__main__":
    main()
