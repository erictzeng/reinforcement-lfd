#!/usr/bin/env python

from __future__ import division
import numpy as np
import scipy.spatial.distance as ssd
import scipy.spatial as sp_spat
from rapprentice.registration import loglinspace, ThinPlateSpline, fit_ThinPlateSpline, tps_reg_cost

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from plotting_plt import plot_warped_grid_2d, plot_warped_grid_3d

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

def sim_annealing_registration(x_nd, y_md, em_step_fcn, n_iter = 20, lambda_init = .1, lambda_final = .001, T_init = .2, T_final = .0002, 
                               plotting = False, plot_cb = None, rot_reg = np.r_[1e-4, 1e-4, 1e-1], beta = 0, vis_cost_xy = None, em_iter = 5):
    """
    Outer loop of simulated annealing
    when em_step_fcn = rpm_em_step, this is tps-rpm algorithm mostly as described by chui and rangaran
    lambda_init/lambda_final: regularization on curvature
    T_init/T_final: radius for correspondence calculation (meters)
    plotting: 0 means don't plot. integer n means plot every n iterations
    vis_cost_xy: matrix of pairwise costs between source and target points, based on visual features
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
            corr_nm, f = em_step_fcn(x_nd, y_md, lambdas[i], Ts[i], rot_reg, f, beta, vis_cost_xy)
        
        if plotting and i%plotting==0:
            plot_cb(x_nd, y_md, corr_nm, f)
    print "Warp cost:", tps_reg_cost(f)
    return f

def rpm_em_step(x_nd, y_md, l, T, rot_reg, prev_f, beta = 0, vis_cost_xy = None, T0 = .2, normalize_iter = 20):
    """
    Function for TPS-RPM (as described in Chui et al.), with and w/o visual
    features.
    """
    xwarped_nd = prev_f.transform_points(x_nd)
    
    dist_nm = ssd.cdist(xwarped_nd, y_md, 'sqeuclidean') / (2*T)
    if beta != 0 and vis_cost_xy:
        dist_nm += beta * vis_cost_xy
    prob_nm = np.exp( -dist_nm ) / np.sqrt(T)
        
    outlier_dist_1m = ssd.cdist(np.mean(xwarped_nd, axis=0)[None,:], y_md, 'sqeuclidean')
    outlier_dist_n1 = ssd.cdist(xwarped_nd, np.mean(y_md, axis=0)[None,:], 'sqeuclidean')
    outlier_prob_1m = np.exp( -outlier_dist_1m / (2*T0) ) / np.sqrt(T0) # add visual cost to outlier terms?
    outlier_prob_n1 = np.exp( -outlier_dist_n1 / (2*T0) ) / np.sqrt(T0)
    
    n,m = prob_nm.shape
    prob_NM = np.empty((n+1, m+1))
    prob_NM[:n, :m] = prob_nm
    prob_NM[:n, m][:,None] = outlier_prob_n1
    prob_NM[n, :m][None,:] = outlier_prob_1m
    prob_NM[n, m] = 0
    
    for _ in xrange(normalize_iter):
        prob_NM /= prob_NM.sum(axis=0)[None,:] # normalize along columns
        prob_NM /= prob_NM.sum(axis=1)[:,None] # normalize along rows
    corr_nm = prob_NM[:n,:m]
    corr_nm += 1e-9 # add noise

    wt_n = corr_nm.sum(axis=1)

    xtarg_nd = (corr_nm/wt_n[:,None]).dot(y_md)

    f = fit_ThinPlateSpline(x_nd, xtarg_nd, bend_coef = l, wt_n = wt_n, rot_coef = rot_reg)
    return corr_nm, f

def reg4_em_step(x_nd, y_md, l, T, rot_reg, prev_f, beta = 0, vis_cost_xy = None, delta = 0.1):
    """
    Function for Reg4 (as described in Combes and Prima), with and w/o visual
    features. Implemented following the pseudocode in "Algo Reg4" exactly.
    delta - cutoff distance for truncated Gaussian
    Note: set delta equal to 2*sqrt(T) or 3*sqrt(T)? Since T = sigma^2
    """
    n, d = x_nd.shape
    m, _ = y_md.shape

    # E-step
    A = np.zeros((m, n))
    kd_y_md = sp_spat.KDTree(y_md)
    for k in range(n):
        x_k = x_nd[k,:]
        x_k_warped = prev_f.transform_points(np.reshape(x_k, (1,d)))[0,:]
        S = kd_y_md.query_ball_point(x_k_warped, delta)  # returns indices
        for j in S:
            y_j = y_md[j,:]
            vis_cost_j_k = 0
            if vis_cost_xy is not None:
                vis_cost_j_k = vis_cost_xy[k, j]
            sq_dist = np.linalg.norm(y_j - x_k_warped)**2
            if sq_dist / (2*T) + beta*vis_cost_j_k <= delta:
                A[j,k] = np.exp(-sq_dist / (2*T) - beta*vis_cost_j_k)
    B = A.copy()

    # Normalize rows of A; normalize columns of B. Compute p_j and q_k
    p = np.zeros(m)
    q = np.zeros(n)
    for j in range(m):
        if sum(A[j,:]) > 0:
            p[j] = 1
            A[j,:] = A[j,:] / float(sum(A[j,:]))
    for k in range(n):
        if sum(B[:,k]) > 0:
            q[k] = 1
            B[:,k] = B[:,k] / float(sum(B[:,k]))

    # Compute A(.k) + B(.k) and y_k
    y_md_approx = np.zeros((n, d))
    wt = np.zeros(n)
    for k in range(n):
        wt[k] = sum(A[:,k] + B[:,k])
        if wt[k] == 0:
            wt[k] = 1e-9  # To avoid division error
        y_md_approx[k,:] = (p*A[:,k] + np.repeat(q[k], m)*B[:,k]).dot(y_md) / float(wt[k])

    # M-step
    f = fit_ThinPlateSpline(x_nd, y_md_approx, bend_coef = l, wt_n = wt, rot_coef = rot_reg)
    return A, f

def reg4_em_step_fast(x_nd, y_md, l, T, rot_reg, prev_f, beta = 0, vis_cost_xy = None, delta = 0.1):
    """
    Function for Reg4 (as described in Combes and Prima), with and w/o visual
    features. Has a few modifications from the pseudocode in "Algo Reg4" exactly.
    delta - cutoff distance for truncated Gaussian
    """
    n, d = x_nd.shape
    m, _ = y_md.shape

    xwarped_nd = prev_f.transform_points(x_nd)
    
    dist_mn = ssd.cdist(y_md, xwarped_nd, 'sqeuclidean') / (2*T)
    if beta != 0 and vis_cost_xy:
        dist_mn += beta * vis_cost_xy.T
    A = np.zeros((m, n))
    A = A.reshape((-1,))
    dist_mn = dist_mn.reshape((-1,))
    A[dist_mn <= delta] = np.exp( -dist_mn )[dist_mn <= delta]
    A = A.reshape((m,n))
    B = A.copy()

    # Normalize rows of A; normalize columns of B. Compute p_j and q_k
    A_rowsum_m = A.sum(axis=1)
    B_colsum_n = B.sum(axis=0)
    p = A_rowsum_m != 0
    q = B_colsum_n != 0

    A[p,:] /= A_rowsum_m[p][:,None] # normalize along rows for non-zero rows
    B[:,q] /= B_colsum_n[q][None,:] # normalize along columns for non-zero columns
    p = p.astype(float)
    q = q.astype(float)
    
    # Compute A(.k) + B(.k) and y_k
    pA_qB = (p[:,None] * A + q[None,:] * B) # are p and q necessary?
    wt = pA_qB.sum(axis=0)
    wt[wt == 0] = 1e-9 # To avoid division error
    y_md_approx = pA_qB.T.dot(y_md) / wt[:,None]
        
    # M-step
    f = fit_ThinPlateSpline(x_nd, y_md_approx, bend_coef = l, wt_n = wt, rot_coef = rot_reg)
    return A, f

def plot_callback(x_nd, y_md, corr_nm, f, x_color=None, y_color=None):
    """
    Plots warp visualization
    x_nd: source points plotted with '+' and x_color (or red if not especified)
    y_md: target points plotted with 'x' and y_color (or blue if not especified)
    warped points plotted with 'o' and x_color (or green if not especified)
    """
    _,d = x_nd.shape
    
    if x_color == None:
        x_color = (1,0,0,1)
        xwarped_color = (0,1,0,1)
    else:
        xwarped_color = x_color
    if y_color == None:
        y_color = (0,0,1,1)
    
    if d == 3:
        plot_callback_3d(x_nd, y_md, corr_nm, f, x_color, y_color, xwarped_color)
    else:
        plot_callback_2d(x_nd, y_md, corr_nm, f, x_color, y_color, xwarped_color)

def plot_callback_2d(x_nd, y_md, corr_nm, f, x_color, y_color, xwarped_color):
    # set interactive
    plt.ion()
    
    # clear previous plots
    plt.clf()
    plt.cla()

    plt.scatter(x_nd[:,0], x_nd[:,1], c=x_color, marker='+', s=50)
    plt.scatter(y_md[:,0], y_md[:,1], c=y_color, marker='x', s=50)
    xwarped_nd = f.transform_points(x_nd)
    plt.scatter(xwarped_nd[:,0], xwarped_nd[:,1], c=xwarped_color, marker='o', s=50)
    
    grid_means = .5 * (x_nd.max(axis=0) + x_nd.min(axis=0))
    grid_mins = grid_means - (x_nd.max(axis=0) - x_nd.min(axis=0))
    grid_maxs = grid_means + (x_nd.max(axis=0) - x_nd.min(axis=0))
    plot_warped_grid_2d(f.transform_points, grid_mins, grid_maxs)
    
    plt.draw()

def plot_callback_3d(x_nd, y_md, corr_nm, f, x_color, y_color, xwarped_color):
    # set interactive
    plt.ion()
    
    # clear previous plots
    plt.clf()
    plt.cla()
    
    ax = plt.gcf().gca(projection='3d')
    ax.set_aspect('equal')

    ax.scatter(x_nd[:,0], x_nd[:,1], x_nd[:,2], c=x_color, marker='+', s=50)
    ax.scatter(y_md[:,0], y_md[:,1], y_md[:,2], c=y_color, marker='x', s=50)
    xwarped_nd = f.transform_points(x_nd)
    ax.scatter(xwarped_nd[:,0], xwarped_nd[:,1], xwarped_nd[:,2], c=xwarped_color, marker='o', s=50)

    # manually set axes limits at a cube's bounding box since matplotlib doesn't correctly set equal axis in 3D
    max_pts = np.r_[x_nd, y_md, xwarped_nd].max(axis=0)
    min_pts = np.r_[x_nd, y_md, xwarped_nd].min(axis=0)
    max_range = (max_pts - min_pts).max()
    center = 0.5*(max_pts + min_pts)
    ax.set_xlim(center[0] - 0.5*max_range, center[0] + 0.5*max_range)
    ax.set_ylim(center[1] - 0.5*max_range, center[1] + 0.5*max_range)
    ax.set_zlim(center[2] - 0.5*max_range, center[2] + 0.5*max_range)

    grid_means = .5 * (x_nd.max(axis=0) + x_nd.min(axis=0))
    grid_mins = grid_means - (x_nd.max(axis=0) - x_nd.min(axis=0))
    grid_maxs = grid_means + (x_nd.max(axis=0) - x_nd.min(axis=0))
    plot_warped_grid_3d(f.transform_points, grid_mins, grid_maxs)
    
    plt.draw()

def main():
    # Test reg4_em_step
    test_x_nd = np.asarray([[1, 1], [1, 2]])
    test_y_md = np.asarray([[2, 2], [2, 3], [2, 4]])
    print "reg4_em_step warps"
    _, test_f = reg4_em_step(test_x_nd, test_y_md, 0.1, 0.1, 1e-3, ThinPlateSpline(2), beta = 0.001, vis_cost_xy = None, delta = 100)
    print "Warp of [1, 1]:", test_f.transform_points(np.asarray([[1,1]]))
    print "Warp of [1, 1.5]:", test_f.transform_points(np.asarray([[1,1.5]]))
    print "Warp of [1, 2]:", test_f.transform_points(np.asarray([[1,2]]))

    print "rpm_em_step warps"
    _, test_f = rpm_em_step(test_x_nd, test_y_md, 0.1, 0.1, 1e-3, ThinPlateSpline(2))
    print "Warp of [1, 1]:", test_f.transform_points(np.asarray([[1,1]]))
    print "Warp of [1, 1.5]:", test_f.transform_points(np.asarray([[1,1.5]]))
    print "Warp of [1, 2]:", test_f.transform_points(np.asarray([[1,2]]))
    
    x_nd = np.c_[np.linspace(-2.5, 2.5, 10), np.linspace(-2.5, 2.5, 10)]
    y_md = np.c_[np.r_[np.linspace(-1.5, 1.5, 7), np.linspace(1.5, 2, 8)], np.linspace(-2.5, 2.5, 15)]
    f = sim_annealing_registration(x_nd, y_md, rpm_em_step, plotting = True, plot_cb = plot_callback)

if __name__ == "__main__":
    main()
