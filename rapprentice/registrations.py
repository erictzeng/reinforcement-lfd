#!/usr/bin/env python

from __future__ import division
import numpy as np
import scipy.spatial.distance as ssd
import scipy.spatial as sp_spat
from rapprentice.registration import loglinspace, ThinPlateSpline, fit_ThinPlateSpline
import IPython as ipy

def sim_annealing_registration(x_nd, y_md, em_step_fcn, n_iter = 20, lambda_init = .1, lambda_final = .001, T_init = .1, T_final = .005, 
                               plotting = False, plot_cb = None, rot_reg = 1e-3, beta = 0, vis_cost = None, em_iter = 5):
    """
    Outer loop of simulated annealing
    when em_step_fcn = rpm_em_step, this is tps-rpm algorithm mostly as described by chui and rangaran
    lambda_init/lambda_final: regularization on curvature
    T_init/T_final: radius for correspondence calculation (meters)
    plotting: 0 means don't plot. integer n means plot every n iterations
    vis_cost: matrix of pairwise costs between source and target points, based on visual features
    """
    _,d=x_nd.shape
    lambdas = loglinspace(lambda_init, lambda_final, n_iter)
    Ts = loglinspace(T_init, T_final, n_iter)

    f = ThinPlateSpline(d) # TODO boxify in here
    f.trans_g = np.median(y_md,axis=0) - np.median(x_nd,axis=0) # align the medians
    
    for i in xrange(n_iter):
        for _ in xrange(em_iter):
            corr_nm, f = em_step_fcn(x_nd, y_md, lambdas[i], Ts[i], rot_reg, f, beta, vis_cost)
        
        if plotting and i%plotting==0:
            plot_cb(x_nd, y_md, corr_nm, f)
    return f

def rpm_em_step(x_nd, y_md, l, T, rot_reg, prev_f, beta = 0, vis_cost = None, T0 = .1, normalize_iter = 20):
    """
    Function for TPS-RPM (as described in Chui et al.), with and w/o visual
    features.
    """
    xwarped_nd = prev_f.transform_points(x_nd)
    
    dist_nm = ssd.cdist(xwarped_nd, y_md, 'euclidean')
    if beta != 0 and vis_cost:
        prob_nm = np.exp( -dist_nm / (2*T) - beta * vis_cost) / T
    else:
        prob_nm = np.exp( -dist_nm / (2*T) ) / T
        
    outlier_dist_1m = ssd.cdist(np.mean(xwarped_nd, axis=0)[None,:], y_md, 'euclidean')
    outlier_dist_n1 = ssd.cdist(xwarped_nd, np.mean(y_md, axis=0)[None,:], 'euclidean')
    outlier_prob_1m = np.exp( -outlier_dist_1m / (2*T0) ) / T0 # add visual cost to outlier terms?
    outlier_prob_n1 = np.exp( -outlier_dist_n1 / (2*T0) ) / T0
    
    n,m = prob_nm.shape
    prob_NM = np.empty((n+1, m+1))
    prob_NM[:n, :m] = prob_nm
    prob_NM[:n, m][:,None] = outlier_prob_n1
    prob_NM[n, :m][None,:] = outlier_prob_1m
    prob_NM[n, m] = 0
    
    for _ in xrange(normalize_iter):
        prob_NM = prob_NM / prob_NM.sum(axis=0)[None,:] # normalize along columns
        prob_NM = prob_NM / prob_NM.sum(axis=1)[:,None] # normalize along rows
    corr_nm = prob_NM[:n,:m]
    corr_nm += 1e-9 # add noise

    wt_n = corr_nm.sum(axis=1)

    xtarg_nd = (corr_nm/wt_n[:,None]).dot(y_md)

    f = fit_ThinPlateSpline(x_nd, xtarg_nd, bend_coef = l, wt_n = wt_n, rot_coef = rot_reg)
    return corr_nm, f

def reg4_em_step(x_nd, y_md, l, T, rot_reg, prev_f, beta = 0, vis_cost = None, delta = 0.1):
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
            if vis_cost:
                vis_cost_j_k = vis_cost(y_j, x_k)
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
        wt[k] = sum(A[:,k]) + sum(B[:,k])
        y_md_approx[k,:] = (p*A[:,k] + np.repeat(q[k], m)*B[:,k]).dot(y_md) / float(wt[k])

    # M-step
    f = fit_ThinPlateSpline(x_nd, y_md_approx, bend_coef = l, wt_n = wt, rot_coef = rot_reg)
    return A, f

def main():
    # Test reg4_em_step
    test_x_nd = np.asarray([[1, 1], [1, 2]])
    test_y_md = np.asarray([[2, 2], [2, 3], [2, 4]])
    _, test_f = reg4_em_step(test_x_nd, test_y_md, 0.1, 0.1, 1e-3, ThinPlateSpline(2), beta = 0.001, vis_cost = None, delta = 100)
    print "reg4_em_step warps"
    print "Warp of [1, 1]:", test_f.transform_points(np.asarray([[1,1]]))
    print "Warp of [1, 1.5]:", test_f.transform_points(np.asarray([[1,1.5]]))
    print "Warp of [1, 2]:", test_f.transform_points(np.asarray([[1,2]]))

    _, test_f = rpm_em_step(test_x_nd, test_y_md, 0.1, 0.1, 1e-3, ThinPlateSpline(2))
    print "rpm_em_step warps"
    print "Warp of [1, 1]:", test_f.transform_points(np.asarray([[1,1]]))
    print "Warp of [1, 1.5]:", test_f.transform_points(np.asarray([[1,1.5]]))
    print "Warp of [1, 2]:", test_f.transform_points(np.asarray([[1,2]]))
    
    test_f = sim_annealing_registration(test_x_nd, test_y_md, rpm_em_step)
    print "sim_annealing_registration rpm_em_step warps"
    print "Warp of [1, 1]:", test_f.transform_points(np.asarray([[1,1]]))
    print "Warp of [1, 1.5]:", test_f.transform_points(np.asarray([[1,1.5]]))
    print "Warp of [1, 2]:", test_f.transform_points(np.asarray([[1,2]]))
    
if __name__ == "__main__":
    main()
