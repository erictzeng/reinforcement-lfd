#!/usr/bin/env python

from __future__ import division
import numpy as np
import scipy.spatial.distance as ssd
from rapprentice.registration import loglinspace, ThinPlateSpline, fit_ThinPlateSpline
import tps

import IPython as ipy

def rgb2lab(rgb):
    return xyz2lab(rgb2xyz(rgb))

def rgb2xyz(rgb):
    """
    r,g,b ranges from 0 to 1
    http://en.wikipedia.org/wiki/SRGB_color_space
    http://en.wikipedia.org/wiki/CIE_XYZ
    """
    rgb_linear = np.empty_like(rgb) # copy rgb so that the original rgb is not modified
    
    cond = rgb > 0.04045
    rgb_linear[cond] = np.power((rgb[cond] + 0.055) / 1.055, 2.4)
    rgb_linear[~cond] = rgb[~cond] / 12.92
    
    rgb_to_xyz = np.array([[0.412453, 0.357580, 0.180423],
                           [0.212671, 0.715160, 0.072169],
                           [0.019334, 0.119193, 0.950227]])
    xyz = rgb_linear.dot(rgb_to_xyz.T)
    return xyz

def xyz2lab(xyz):
    """
    l ranges from 0 to 100 and a,b ranges from -128 to 128
    http://en.wikipedia.org/wiki/Lab_color_space
    """
    ref = np.array([0.95047, 1., 1.08883]) # CIE LAB constants for Observer = 2deg, Illuminant = D65
    xyz = xyz / ref # copy xyz so that the original xyz is not modified

    cond = xyz > 0.008856
    xyz[cond] = np.power(xyz[cond], 1./3.)
    xyz[~cond] = 7.787 * xyz[~cond] + 16./116.
    
    x,y,z = xyz.T
    l = 116. * y - 16.
    a = 500. * (x - y)
    b = 200. * (y - z)
    
    lab = np.array([l,a,b]).T
    return lab

def ab_cost(xyzrgb1, xyzrgb2):
    _,d = xyzrgb1.shape
    d -= 3  # subtract out the three RGB coordinates
    lab1 = rgb2lab(xyzrgb1[:,d:])
    lab2 = rgb2lab(xyzrgb2[:,d:])
    cost = ssd.cdist(lab1[:,1:], lab2[:,1:], 'euclidean')
    return cost

def sinkhorn_balance_coeffs(prob_NM, normalize_iter):
    """
    Computes the coefficients to balance the matrix prob_NM. Unlike balance_matrix2, row-normalization happens first.
    The coefficients are computed with type 'f4', so it's better if prob_NM is already in type 'f4'.
    The sinkhorn_balance_matrix can be then computed in the following way:
    prob_NM *= r_N[:,None]
    prob_NM *= c_M[None,:]
    """
    if prob_NM.dtype != np.dtype('f4'):
        prob_NM = prob_NM.astype('f4')
    N,M = prob_NM.shape
    c_M = np.ones(M,'f4')
    for _ in xrange(normalize_iter):
        r_N = 1./prob_NM.dot(c_M) # normalize along rows
        c_M = 1./r_N.dot(prob_NM) # normalize along columns
    return r_N, c_M

def tps_rpm(x_nd, y_md, n_iter = 20, lambda_init = 10., lambda_final = .1, T_init = .04, T_final = .00004, rot_reg = np.r_[1e-4, 1e-4, 1e-1], 
            plotting = False, plot_cb = None, outlierfrac = 1e-2, vis_cost_xy = None, em_iter = 2):
    """
    tps-rpm algorithm mostly as described by chui and rangaran
    lambda_init/lambda_final: regularization on curvature
    T_init/T_final: radius for correspondence calculation (meters)
    plotting: 0 means don't plot. integer n means plot every n iterations
    vis_cost_xy: matrix of pairwise costs between source and target points, based on visual features
    Note: Pick a T_init that is about 1/10 of the largest square distance of all point pairs
    """
    _,d=x_nd.shape
    lambdas = loglinspace(lambda_init, lambda_final, n_iter)
    Ts = loglinspace(T_init, T_final, n_iter)

    f = ThinPlateSpline(d)
    scale = (np.max(y_md,axis=0) - np.min(y_md,axis=0)) / (np.max(x_nd,axis=0) - np.min(x_nd,axis=0))
    f.lin_ag = np.diag(scale).T # align the mins and max
    f.trans_g = np.median(y_md,axis=0) - np.median(x_nd,axis=0) * scale  # align the medians

    for i in xrange(n_iter):
        for _ in xrange(em_iter):
            f, corr_nm = rpm_em_step(x_nd, y_md, lambdas[i], Ts[i], rot_reg, f, vis_cost_xy = vis_cost_xy)

        if plotting and (i%plotting==0 or i==(n_iter-1)):
            plot_cb(x_nd, y_md, corr_nm, f, i)
    return f, corr_nm

def rpm_em_step(x_nd, y_md, l, T, rot_reg, prev_f, vis_cost_xy = None, outlierfrac = 1e-2, normalize_iter = 20):
    n,d = x_nd.shape
    m,_ = y_md.shape
    xwarped_nd = prev_f.transform_points(x_nd)
    
    dist_nm = ssd.cdist(xwarped_nd, y_md, 'sqeuclidean')
    prob_nm = np.exp( -dist_nm / (2*T) )
    if vis_cost_xy != None:
        pi = np.exp( -vis_cost_xy )
        prob_nm *= pi
    prob_nm /= prob_nm.sum(axis=0)[None,:] # normalize along columns; these are proper probabilities over j = 1,...,N
    prob_nm = np.maximum(prob_nm, (1e-40)*np.ones((n,m))) # clamp at 1e-40 to prevent nans in the normalization. TODO figure out a better way to handle these cases

    outlier_prob_1m = outlierfrac/(1.-outlierfrac) * np.ones((1,m)) # so that outlier_prob_1m[0,:] / (prob_nm.sum(axis=0) + outlier_prob_1m[0,:]) = outlierfrac * np.ones(m)
    outlier_prob_n1 = outlierfrac * prob_nm.sum(axis=1)[:,None]/(1. - outlierfrac) # so that outlier_prob_n1[:,0] / (prob_nm.sum(axis=1) + outlier_prob_n1[:,0]) = outlierfrac * np.ones(n)

    prob_NM = np.empty((n+1, m+1), 'f4')
    prob_NM[:n, :m] = prob_nm
    prob_NM[:n, m][:,None] = outlier_prob_n1
    prob_NM[n, :m][None,:] = outlier_prob_1m
    prob_NM[n, m] = 0

    r_N, c_M = sinkhorn_balance_coeffs(prob_NM, normalize_iter)
    prob_NM *= r_N[:,None]
    prob_NM *= c_M[None,:]
    corr_nm = prob_NM[:n,:m]

    wt_n = corr_nm.sum(axis=1)

    xtarg_nd = (corr_nm/wt_n[:,None]).dot(y_md)

    f = fit_ThinPlateSpline(x_nd, xtarg_nd, bend_coef = l, wt_n = wt_n, rot_coef = rot_reg)
    f._bend_coef = l
    f._rot_coef = rot_reg
    f._cost = tps.tps_cost(f.lin_ag, f.trans_g, f.w_ng, f.x_na, xtarg_nd, l, wt_n=wt_n)/wt_n.mean()

    return f, corr_nm

def main():
    import argparse, h5py, os
    import matplotlib.pyplot as plt
    from rapprentice import clouds, plotting_plt
    import registration
    import time
    
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("--output_folder", type=str, default="")
    parser.add_argument("--plot_color", type=int, default=1)
    parser.add_argument("--proj", type=int, default=1, help="project 3d visualization into 2d")
    parser.add_argument("--visual_prior", type=int, default=1)
    parser.add_argument("--plotting", type=int, default=1)

    args = parser.parse_args()
    
    # TODO use PCL's downsampling for color as well
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
                plotting_plt.plot_tps_registration(x_nd, y_md, f, x_color = x_color, y_color = y_color, proj_2d=args.proj)
            else:
                plotting_plt.plot_tps_registration(x_nd, y_md, f, proj_2d=args.proj)
            # save plot to file
            if output_prefix is not None:
                plt.savefig(output_prefix + "_iter" + str(iteration) + '.png')
        return plot_cb

    def plot_cb_bij_gen(output_prefix, args, x_color, y_color):
        def plot_cb_bij(x_nd, y_md, xtarg_nd, corr_nm, wt_n, f):
            if args.plot_color:
                plotting_plt.plot_tps_registration(x_nd, y_md, f, res = (.3, .3, .12), x_color = x_color, y_color = y_color, proj_2d=args.proj)
            else:
                plotting_plt.plot_tps_registration(x_nd, y_md, f, res = (.4, .3, .12), proj_2d=args.proj)
            # save plot to file
            if output_prefix is not None:
                plt.savefig(output_prefix + "_iter" + str(iteration) + '.png')
        return plot_cb_bij

    # preprocess and downsample clouds
    infile = h5py.File(args.input_file)
    source_clouds = {}
    target_clouds = {}
    for i in range(len(infile)):
        source_cloud = downsample_cloud(infile[str(i)]['source_cloud'][()])
        source_clouds[i] = source_cloud
        target_clouds[i] = []
        for (cloud_key, target_cloud) in infile[str(i)]['target_clouds'].iteritems():
            target_cloud = downsample_cloud(target_cloud[()])
            target_clouds[i].append(target_cloud)
    infile.close()
    
    start_time = time.time()
    rpm_tps_costs = []
    rpm_tps_reg_cost = []
    for i in range(len(source_clouds)):
        source_cloud = source_clouds[i]
        for target_cloud in target_clouds[i]:
            if args.visual_prior:
                vis_cost_xy = ab_cost(source_cloud, target_cloud)
            else:
                vis_cost_xy = None
            f, corr_nm = tps_rpm(source_cloud[:,:-3], target_cloud[:,:-3],
                                 vis_cost_xy = vis_cost_xy,
                                 plotting=args.plotting, plot_cb = plot_cb_gen(os.path.join(args.output_folder, str(i) + "_" + cloud_key + "_rpm") if args.output_folder else None,
                                                                               args,
                                                                               source_cloud[:,-3:],
                                                                               target_cloud[:,-3:]))
            rpm_tps_costs.append(f._cost)
            rpm_tps_reg_cost.append(registration.tps_reg_cost(f))
    print "tps_rpm time elapsed", time.time() - start_time
    
    start_time = time.time()
    rpm_bij_tps_costs = []
    rpm_bij_tps_reg_cost = []
    for i in range(len(source_clouds)):
        source_cloud = source_clouds[i]
        for target_cloud in target_clouds[i]:
            if args.visual_prior:
                vis_cost_xy = ab_cost(source_cloud, target_cloud)
            else:
                vis_cost_xy = None
            x_nd = source_cloud[:,:3]
            y_md = target_cloud[:,:3]
            scaled_x_nd, _ = registration.unit_boxify(x_nd)
            scaled_y_md, _ = registration.unit_boxify(y_md)
            f,g = registration.tps_rpm_bij(scaled_x_nd, scaled_y_md, rot_reg=np.r_[1e-4, 1e-4, 1e-1], n_iter=50, reg_init=10, reg_final=.1, outlierfrac=1e-2, vis_cost_xy=vis_cost_xy,
                                           plotting=args.plotting, plot_cb=plot_cb_bij_gen(os.path.join(args.output_folder, str(i) + "_" + cloud_key + "_rpm_bij") if args.output_folder else None,
                                                                                           args,
                                                                                           source_cloud[:,-3:],
                                                                                           target_cloud[:,-3:]))
            rpm_bij_tps_costs.append(f._cost)
            rpm_bij_tps_reg_cost.append(registration.tps_reg_cost(f))
    print "tps_rpm_bij time elapsed", time.time() - start_time
    
    start_time = time.time()
    rpm_bij_cheap_tps_costs = []
    rpm_bij_cheap_tps_reg_cost = []
    for i in range(len(source_clouds)):
        source_cloud = source_clouds[i]
        for target_cloud in target_clouds[i]:
            if args.visual_prior:
                vis_cost_xy = ab_cost(source_cloud, target_cloud)
            else:
                vis_cost_xy = None
            x_nd = source_cloud[:,:3]
            y_md = target_cloud[:,:3]
            scaled_x_nd, _ = registration.unit_boxify(x_nd)
            scaled_y_md, _ = registration.unit_boxify(y_md)
            f,g = registration.tps_rpm_bij(scaled_x_nd, scaled_y_md, rot_reg=np.r_[1e-4, 1e-4, 1e-1], n_iter=10, vis_cost_xy=vis_cost_xy, # Note registration_cost_cheap in rope_qlearn has a different rot_reg
                                           plotting=args.plotting, plot_cb=plot_cb_bij_gen(os.path.join(args.output_folder, str(i) + "_" + cloud_key + "_rpm_bij_cheap") if args.output_folder else None,
                                                                                           args,
                                                                                           source_cloud[:,-3:],
                                                                                           target_cloud[:,-3:]))
            rpm_bij_cheap_tps_costs.append(f._cost)
            rpm_bij_cheap_tps_reg_cost.append(registration.tps_reg_cost(f))
    print "tps_rpm_bij_cheap time elapsed", time.time() - start_time
    
    np.set_printoptions(suppress=True)
    
    print ""
    print "tps_costs"
    print "rpm, bij, bij_cheap"
    print np.array([rpm_tps_costs, rpm_bij_tps_costs, rpm_bij_cheap_tps_costs]).T
    
    print ""
    print "tps_reg_cost"
    print "rpm, bij, bij_cheap"
    print np.array([rpm_tps_reg_cost, rpm_bij_tps_reg_cost, rpm_bij_cheap_tps_reg_cost]).T

if __name__ == "__main__":
    main()
