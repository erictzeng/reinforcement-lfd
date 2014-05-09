#!/usr/bin/env python

import numpy as np
import scipy.spatial as sp_spat
from rapprentice.registration import fit_ThinPlateSpline, ThinPlateSpline

# Function containing simulated annealing outer loop
# Parameters:
#     em_step_fcn
#     x_nd
#     y_md
#     n_iter
#     lambda
#     T_init
#     T_final
#     rot_reg
#     vis_cost


# Function for TPS-RPM (as described in Chui et al.), with and w/o visual
# features.
# Parameters:
#     x_nd
#     y_md
#     lambda
#     T
#     rot_reg
#     beta
#     vis_cost - function that returns cost btwn two points, based on visual features


# Function for Reg4 (as described in Combes and Prima), with and w/o visual
# features. Implemented following the pseudocode in "Algo Reg4" exactly.
# Parameters:
#     x_nd
#     y_md
#     lambda
#     T
#     rot_reg
#     prev_f - rapprentice.registration.ThinPlateSpline object
#     beta
#     vis_cost
#     delta - cutoff distance for truncated Gaussian
# Note: set delta equal to 2*sqrt(T) or 3*sqrt(T)? Since T = sigma^2
def reg4_em_step(x_nd, y_md, l, T, rot_reg, prev_f, beta, vis_cost = None, delta = 0.1):
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
                A[j,k] = np.exp(-1*sq_dist - beta*vis_cost_j_k)
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
    print A
    print B
    for k in range(n):
        wt[k] = sum(A[:,k]) + sum(B[:,k])
        y_md_approx[k,:] = (p*A[:,k] + np.repeat(q[k], m)*B[:,k]).dot(y_md) / float(wt[k])

    # M-step
    f = fit_ThinPlateSpline(x_nd, y_md_approx, bend_coef = l, wt_n = wt, rot_coef = rot_reg)
    return f

def main():
    # Test reg4_em_step
    test_x_nd = np.asarray([[1, 1], [1, 2]])
    test_y_md = np.asarray([[2, 2], [2, 3], [2, 4]])
    test_f = reg4_em_step(test_x_nd, test_y_md, 0.1, 0.1, 1e-3, ThinPlateSpline(2), beta = 0.001, vis_cost = None, delta = 100)
    print "Warp of [1, 1]:", test_f.transform_points(np.asarray([[1,1]]))
    print "Warp of [1, 1.5]:", test_f.transform_points(np.asarray([[1,1.5]]))
    print "Warp of [1, 2]:", test_f.transform_points(np.asarray([[1,2]]))

if __name__ == "__main__":
    main()
