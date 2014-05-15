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
    return cloud_xyzrgb_downsamp

def plot_cb_gen(output_prefix, args, x_color, y_color):
    def plot_cb(x_nd, y_md, corr_nm, f, iteration):
        if args.plot_color:
            registrations.plot_callback(x_nd, y_md, corr_nm, f, iteration, output_prefix, x_color = x_color, y_color = y_color, proj_2d=args.proj)
        else:
            registrations.plot_callback(x_nd, y_md, corr_nm, f, iteration, output_prefix, proj_2d=args.proj)
    return plot_cb

def run_experiments(args):
    if 0 in args.experiments:
        run_experiments0(args)
    if 1 in args.experiments:
        run_experiments1(args)

def run_experiments1(args):
    infile = h5py.File(args.input_file)
    for i in range(len(infile)):
        source_cloud = downsample_cloud(infile[str(i)]['source_cloud'][()])
        
        for (cloud_key, target_cloud) in infile[str(i)]['target_clouds'].iteritems():
            target_cloud = downsample_cloud(target_cloud[()])
            print "source cloud %d and target cloud %s"%(i, cloud_key)
            
            vis_costs_xy = registrations.ab_cost(source_cloud, target_cloud)
    
            f, bend_cost, res_cost, total_cost = registrations.sim_annealing_registration(source_cloud[:,:-3], target_cloud[:,:-3],
                registrations.rpm_em_step, vis_cost_xy = vis_costs_xy,
                plotting=1, plot_cb = plot_cb_gen(os.path.join(args.output_folder, k + "_" + cloud_key + "_rpmvis") if args.output_folder else None,
                                                  args,
                                                  source_cloud[:,-3:],
                                                  target_cloud[:,-3:]))

def run_experiments0(args):
    clouds = h5py.File(args.input_file)
    _,d = clouds['dem_cloud'][()].shape
    d = d - 3  # ignore the RGB values
    x_xyzrgb = downsample_cloud(clouds['dem_cloud'][()])
    x_nd = x_xyzrgb[:,:d]
    if args.output_folder:
        output_prefix = os.path.join(args.output_folder, "")
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

        print "Reg4 EM, w/ visual features"
        f, bend_cost, res_cost, total_cost = registrations.sim_annealing_registration(x_nd, y_md,
                registrations.reg4_em_step, vis_cost_xy = vis_costs_xy,
                plotting=1, plot_cb = plot_cb_gen(os.path.join(args.output_folder, k + "_reg4vis") if args.output_folder else None,
                                                  args,
                                                  x_xyzrgb[:,-3:],
                                                  y_xyzrgb[:,-3:]))
        if rot_degrees > 0:
            if "reg4vis" not in costs:
                costs['reg4vis'] = []
            costs['reg4vis'].append((rot_degrees, bend_cost, res_cost, total_cost))

        print "Reg4 EM, w/o visual features"
        f, bend_cost, res_cost, total_cost = registrations.sim_annealing_registration(x_nd, y_md,
                registrations.reg4_em_step,
                plotting=1, plot_cb = plot_cb_gen(os.path.join(args.output_folder, k + "_reg4vis") if args.output_folder else None,
                                                  args,
                                                  x_xyzrgb[:,-3:],
                                                  y_xyzrgb[:,-3:]))
        if rot_degrees > 0:
            if "reg4" not in costs:
                costs['reg4'] = []
            costs['reg4'].append((rot_degrees, bend_cost, res_cost, total_cost))

        print "RPM EM, w/ visual features"
        f, bend_cost, res_cost, total_cost = registrations.sim_annealing_registration(x_nd, y_md,
                registrations.rpm_em_step, vis_cost_xy = vis_costs_xy,
                plotting=1, plot_cb = plot_cb_gen(os.path.join(args.output_folder, k + "_reg4vis") if args.output_folder else None,
                                                  args,
                                                  x_xyzrgb[:,-3:],
                                                  y_xyzrgb[:,-3:]))
        if rot_degrees > 0:
            if "rpmvis" not in costs:
                costs['rpmvis'] = []
            costs['rpmvis'].append((rot_degrees, bend_cost, res_cost, total_cost))

        print "RPM EM, w/o visual features"
        f, bend_cost, res_cost, total_cost = registrations.sim_annealing_registration(x_nd, y_md,
                registrations.rpm_em_step,
                plotting=1, plot_cb = plot_cb_gen(os.path.join(args.output_folder, k + "_reg4vis") if args.output_folder else None,
                                                  args,
                                                  x_xyzrgb[:,-3:],
                                                  y_xyzrgb[:,-3:]))
        if rot_degrees > 0:
            if "rpm" not in costs:
                costs['rpm'] = []
            costs['rpm'].append((rot_degrees, bend_cost, res_cost, total_cost))
        
#         def plot_cb_bij(x_nd, y_md, xtarg_nd, corr_nm, wt_n, f):
#             iteration = 0 # TODO
#             if args.plot_color:
#                 registrations.plot_callback(x_nd, y_md, corr_nm, f, iteration, os.path.join(args.output_folder, k + "_rpmbij"), res = (.3, .3, .12), x_color = x_xyzrgb[:,d:], y_color = y_xyzrgb[:,d:], proj_2d=args.proj)
#             else:
#                 registrations.plot_callback(x_nd, y_md, corr_nm, f, iteration, os.path.join(args.output_folder, k + "_rpmbij"), res = (.4, .3, .12), proj_2d=args.proj)
#         scaled_x_nd, _ = registration.unit_boxify(x_nd)
#         scaled_y_md, _ = registration.unit_boxify(y_md)
#         f,g = registration.tps_rpm_bij(scaled_x_nd, scaled_y_md, plot_cb=plot_cb_bij,
#                                        plotting=1, rot_reg=np.r_[1e-4, 1e-4, 1e-1][:d], 
#                                        n_iter=50, reg_init=10, reg_final=.1, outlierfrac=1e-2)
        for k in costs:
            cost_vec = costs[k]
            print k
            cost_vec = sorted(cost_vec, key=lambda x: x[0])
            for x in cost_vec:
                print '{} \t {} \t {} \t {}'.format(x[0], x[1], x[2], x[3])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("--experiments", type=int, nargs='*', metavar="i_exp", default=[0])
    parser.add_argument("--output_folder", type=str, default="")
    parser.add_argument("--plot_color", type=int, default=1)
    parser.add_argument("--proj", type=int, default=1, help="project 3d visualization into 2d")

    args = parser.parse_args()
    run_experiments(args)

if __name__ == "__main__":
    main()
