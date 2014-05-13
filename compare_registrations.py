#!/usr/bin/env python

# Conducts experiments for comparing TPS-RPM and Reg4, with and without visual
# features.
#
# Arguments:
#     input_file: h5 file with the key "dem_cloud" corresponding to a numpy
#                 array containing [x,y,r,b,g] info for the original points, and
#                 all other keys corresponding to warped point clouds in the
#                 same format

import argparse, h5py, numpy as np, os
import scipy.spatial.distance as ssd
from rapprentice import clouds, registration, registrations

DS_SIZE = 0.025

def downsample_cloud(cloud):
    # cloud should have XYZRGB info per row
    d = 3
    cloud_xyz = cloud[:,:d]
    cloud_xyz_downsamp = clouds.downsample(cloud_xyz, DS_SIZE)
    new_n,_ = cloud_xyz_downsamp.shape
    dists = ssd.cdist(cloud_xyz_downsamp, cloud_xyz)
    min_indices = dists.argmin(axis=1)
    cloud_xyzrgb_downsamp = np.zeros((new_n,d+3))
    cloud_xyzrgb_downsamp[:,:d] = cloud_xyz_downsamp
    cloud_xyzrgb_downsamp[:,d:] = cloud[min_indices,d:]
    print cloud_xyzrgb_downsamp.shape
    return cloud_xyzrgb_downsamp

def run_experiments(input_file, output_folder, plot_color):
    clouds = h5py.File(input_file)
    _,d = clouds['dem_cloud'][()].shape
    d = d - 3  # ignore the RGB values
    x_xyzrgb = downsample_cloud(clouds['dem_cloud'][()])
    x_nd = x_xyzrgb[:,:d]
    if output_folder:
        output_prefix = os.path.join(output_folder, "")
    else:
        output_prefix = None
    costs = {}

    for k in clouds:
        if not k.startswith('warp'):
            continue

        rot_degrees = 0
        if k.find('warp_rot') > 0 and k.find('test') < 0:
            rot_degrees = int(k.split('rot')[1])

        print "TESTING:", k
        y_xyzrgb = downsample_cloud(clouds[k][()])
        y_md = y_xyzrgb[:,:d]
        vis_costs_xy = registrations.ab_cost(x_xyzrgb, y_xyzrgb)

        def plot_cb_gen(output_prefix):
            def plot_cb(x_nd, y_md, corr_nm, f, iteration):
                if plot_color:
                    registrations.plot_callback(x_nd, y_md, corr_nm, f, iteration, output_prefix, x_color = x_xyzrgb[:,d:], y_color = y_xyzrgb[:,d:])
                else:
                    registrations.plot_callback(x_nd, y_md, corr_nm, f, iteration, output_prefix)
            return plot_cb

        print "Reg4 EM, w/ visual features"
        f, bend_cost, res_cost, total_cost = registrations.sim_annealing_registration(x_nd, y_md,
                registrations.reg4_em_step, vis_cost_xy = vis_costs_xy,
                plotting=1, plot_cb = plot_cb_gen(output_prefix + k + "_reg4vis" if output_prefix else None))
        if rot_degrees > 0:
            if "reg4vis" not in costs:
                costs['reg4vis'] = []
            costs['reg4vis'].append((rot_degrees, bend_cost, res_cost, total_cost))

        print "Reg4 EM, w/o visual features"
        f, bend_cost, res_cost, total_cost = registrations.sim_annealing_registration(x_nd, y_md,
                registrations.reg4_em_step,
                plotting=1, plot_cb = plot_cb_gen(output_prefix + k + "_reg4" if output_prefix else None))
        if rot_degrees > 0:
            if "reg4" not in costs:
                costs['reg4'] = []
            costs['reg4'].append((rot_degrees, bend_cost, res_cost, total_cost))

        print "RPM EM, w/ visual features"
        f, bend_cost, res_cost, total_cost = registrations.sim_annealing_registration(x_nd, y_md,
                registrations.rpm_em_step, vis_cost_xy = vis_costs_xy,
                plotting=1, plot_cb = plot_cb_gen(output_prefixk + "_rpmvis" if output_prefix else None))
        if rot_degrees > 0:
            if "rpmvis" not in costs:
                costs['rpmvis'] = []
            costs['rpmvis'].append((rot_degrees, bend_cost, res_cost, total_cost))

        print "RPM EM, w/o visual features"
        f, bend_cost, res_cost, total_cost = registrations.sim_annealing_registration(x_nd, y_md,
                registrations.rpm_em_step,
                plotting=1, plot_cb = plot_cb_gen(output_prefix + k + "_rpm" if output_prefix else None))
        if rot_degrees > 0:
            if "rpm" not in costs:
                costs['rpm'] = []
            costs['rpm'].append((rot_degrees, bend_cost, res_cost, total_cost))
        
        def plot_cb_bij(x_nd, y_md, xtarg_nd, corr_nm, wt_n, f):
            if plot_color:
                registrations.plot_callback(x_nd, y_md, corr_nm, f, output_prefix + k + "_rpmbij", iteration, res = (.3, .3, .12), x_color = x_xyzrgb[:,d:], y_color = y_xyzrgb[:,d:])
            else:
                registrations.plot_callback(x_nd, y_md, corr_nm, f, output_prefix + k + "_rpmbij", iteration, res = (.4, .3, .12))
        #scaled_x_nd, _ = registration.unit_boxify(x_nd)
        #scaled_y_md, _ = registration.unit_boxify(y_md)
        #f,g = registration.tps_rpm_bij(scaled_x_nd, scaled_y_md, plot_cb=plot_cb_bij,
        #                               plotting=1, rot_reg=np.r_[1e-4, 1e-4, 1e-1][:d], 
        #                               n_iter=50, reg_init=10, reg_final=.1, outlierfrac=1e-2)
        for k in costs:
            cost_vec = costs[k]
            print k
            cost_vec = sorted(cost_vec, key=lambda x: x[0])
            for x in cost_vec:
                print '{} \t {} \t {} \t {}'.format(x[0], x[1], x[2], x[3])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("--output_folder", type=str, default="")
    parser.add_argument("--plot_color", type=int, default=1)

    args = parser.parse_args()
    run_experiments(args.input_file, args.output_folder, args.plot_color)

if __name__ == "__main__":
    main()
