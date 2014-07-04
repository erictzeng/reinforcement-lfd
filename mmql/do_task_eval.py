#!/usr/bin/env python

from __future__ import division

import pprint
import argparse
from ropesimulation import sim_util, transfer_simulate
from rapprentice import eval_util, util
from rapprentice import tps_registration, planning
 
from rapprentice import registration, berkeley_pr2, \
     animate_traj, ros2rave, plotting_openrave, task_execution, \
     tps, func_utils, resampling, ropesim, rope_initialization, clouds
from rapprentice import math_utils as mu
from rapprentice.yes_or_no import yes_or_no
import pdb, time

from constants import ROPE_RADIUS, ROPE_ANG_STIFFNESS, ROPE_ANG_DAMPING, ROPE_LIN_DAMPING, ROPE_ANG_LIMIT,\
    ROPE_LIN_STOP_ERP, ROPE_MASS, ROPE_RADIUS_THICK, DS_SIZE, COLLISION_DIST_THRESHOLD, EXACT_LAMBDA, \
    N_ITER_EXACT, BEND_COEF_DIGITS, MAX_CLD_SIZE, JOINT_LENGTH_PER_STEP, FINGER_CLOSE_RATE

from tpsopt.registration import tps_rpm_bij
from tpsopt.transformations import TPSSolver, EmptySolver
from tpsopt.precompute import loglinspace

from features import BatchRCFeats
import trajoptpy, openravepy
from ropesimulation.rope_utils import get_closing_pts, get_closing_inds
from rapprentice.knot_classifier import isKnot as is_knot, calculateCrossings
import os, os.path, numpy as np, h5py
from numpy import asarray
from rapprentice.util import redprint, yellowprint
import atexit
import importlib
from itertools import combinations
import IPython as ipy
import random
import hashlib

class GlobalVars:
    unique_id = 0
    actions = None
    actions_cache = None
    tps_errors_top10 = []
    trajopt_errors_top10 = []
    actions_ds_clouds = {}
    rope_nodes_crossing_info = {}
    features = None
    action_solvers = {}
    empty_solver = None

# @profile
def register_tps(sim_env, state, action, args_eval, interest_pts = None, closing_hmats = None, 
                 closing_finger_pts = None, reg_type='bij'):
    scaled_x_nd, src_params = get_scaled_action_cloud(state, action)
    new_cloud = state.cloud
    if reg_type == 'bij':
        vis_cost_xy = tps_registration.ab_cost(old_cloud, new_cloud) if args_eval.use_color else None
        y_md = new_cloud[:,:3]
        scaled_y_md, targ_params = registration.unit_boxify(y_md)
        scaled_K_mm = tps.tps_kernel_matrix(scaled_y_md)
        bend_coefs = np.around(loglinspace(EXACT_LAMBDA[0], EXACT_LAMBDA[1], N_ITER_EXACT), BEND_COEF_DIGITS)
        gsolve = GlobalVars.empty_solver.get_solver(scaled_y_md, scaled_K_mm, bend_coefs)
        fsolve = GlobalVars.action_solvers[action]
        rot_reg = np.r_[1e-4, 1e-4, 1e-1]
        reg_init = EXACT_LAMBDA[0]
        reg_final = EXACT_LAMBDA[1]
        n_iter = N_ITER_EXACT
        (f,g), corr = tps_rpm_bij(scaled_x_nd, scaled_y_md, fsolve, gsolve, rot_reg=rot_reg, n_iter=n_iter, 
                                  reg_init=reg_init, reg_final=reg_final, outlierfrac=1e-2, vis_cost_xy=vis_cost_xy, 
                                  return_corr=True, check_solver=True)
        f = registration.unscale_tps(f, src_params, targ_params)
        f._bend_coef = reg_final
        f._rot_coef = rot_reg
        f._wt_n = corr.sum(axis=1)
    else:
        raise RuntimeError('invalid registration type')
    return f, corr
# @profile
def compute_trans_traj(sim_env, state, action, args_eval, transferopt=None, animate=False, interactive=False, simulate=True, replay_full_trajs=None):
    alpha = args_eval.alpha
    beta_pos = args_eval.beta_pos
    beta_rot = args_eval.beta_rot
    gamma = args_eval.gamma
    if transferopt is None:
        transferopt = args_eval.transferopt
    
    seg_info = GlobalVars.actions[action]
    if simulate:
        sim_util.reset_arms_to_side(sim_env)
    
    cloud_dim = 6 if args_eval.use_color else 3
    old_cloud = get_action_cloud_ds(sim_env, action, args_eval)[:,:cloud_dim]
    old_rope_nodes = get_action_rope_nodes(sim_env, action, args_eval)
    
    new_cloud = state.cloud
    new_cloud = new_cloud[:,:cloud_dim]
    
    sim_env.set_rope_state(state)

    handles = []
    if animate:
        # color code: r demo, y transformed, g transformed resampled, b new
        handles.append(sim_env.env.plot3(old_cloud[:,:3], 2, (1,0,0)))
        handles.append(sim_env.env.plot3(new_cloud[:,:3], 2, new_cloud[:,3:] if args_eval.use_color else (0,0,1)))
        sim_env.viewer.Step()
    
    closing_inds = get_closing_inds(seg_info)
    closing_hmats = {}
    for lr in closing_inds:
        if closing_inds[lr] != -1:
            closing_hmats[lr] = seg_info["%s_gripper_tool_frame"%lr]['hmat'][closing_inds[lr]]
    
    miniseg_intervals = []
    for lr in 'lr':
        miniseg_intervals.extend([(i_miniseg_lr, lr, i_start, i_end) for (i_miniseg_lr, (i_start, i_end)) in enumerate(zip(*sim_util.split_trajectory_by_lr_gripper(seg_info, lr)))])
    # sort by the start of the trajectory, then by the length (if both trajectories start at the same time, the shorter one should go first), and then break ties by executing the right trajectory first
    miniseg_intervals = sorted(miniseg_intervals, key=lambda (i_miniseg_lr, lr, i_start, i_end): (i_start, i_end-i_start, {'l':'r', 'r':'l'}[lr]))
    
    miniseg_interval_groups = []
    for (curr_miniseg_interval, next_miniseg_interval) in zip(miniseg_intervals[:-1], miniseg_intervals[1:]):
        curr_i_miniseg_lr, curr_lr, curr_i_start, curr_i_end = curr_miniseg_interval
        next_i_miniseg_lr, next_lr, next_i_start, next_i_end = next_miniseg_interval
        if len(miniseg_interval_groups) > 0 and curr_miniseg_interval in miniseg_interval_groups[-1]:
            continue
        curr_gripper_open = sim_util.binarize_gripper(seg_info["%s_gripper_joint"%curr_lr][curr_i_end])
        next_gripper_open = sim_util.binarize_gripper(seg_info["%s_gripper_joint"%next_lr][next_i_end])
        miniseg_interval_group = [curr_miniseg_interval]
        if not curr_gripper_open and not next_gripper_open and curr_lr != next_lr and curr_i_start < next_i_end and next_i_start < curr_i_end:
            miniseg_interval_group.append(next_miniseg_interval)
        miniseg_interval_groups.append(miniseg_interval_group)
    
    success = True
    feasible = True
    misgrasp = False
    full_trajs = []
    obj_values = []
    for i_miniseg_group, miniseg_interval_group in enumerate(miniseg_interval_groups):
        if not simulate or replay_full_trajs is None: # we are not simulating, we still want to compute the costs
            group_full_trajs = []
            for (i_miniseg_lr, lr, i_start, i_end) in miniseg_interval_group:
                manip_name = {"l":"leftarm", "r":"rightarm"}[lr]                 
                ee_link_name = "%s_gripper_tool_frame"%lr
        
                ################################    
                redprint("Generating %s arm joint trajectory for part %i"%(lr, i_miniseg_lr))
                
                # figure out how we're gonna resample stuff
                old_arm_traj = asarray(seg_info[manip_name][i_start - int(i_start > 0):i_end+1])
                if not sim_util.arm_moved(old_arm_traj):
                    continue
                old_finger_traj = sim_util.gripper_joint2gripper_l_finger_joint_values(seg_info['%s_gripper_joint'%lr][i_start - int(i_start > 0):i_end+1])[:,None]
                _, timesteps_rs = sim_util.unif_resample(old_arm_traj, JOINT_LENGTH_PER_STEP)
            
                ### Generate fullbody traj
                old_arm_traj_rs = mu.interp2d(timesteps_rs, np.arange(len(old_arm_traj)), old_arm_traj)

                f, corr = register_tps(sim_env, state, action, args_eval, reg_type='bij')
                if f is None: break

                if animate:
                    handles.append(sim_env.env.plot3(f.transform_points(old_cloud[:,:3]), 2, old_cloud[:,3:] if args_eval.use_color else (1,1,0)))
                    new_cloud_rs = corr.dot(new_cloud)
                    handles.append(sim_env.env.plot3(new_cloud_rs[:,:3], 2, new_cloud_rs[:,3:] if args_eval.use_color else (0,1,0)))
                    handles.extend(sim_util.draw_grid(sim_env, old_cloud[:,:3], f))
                
                x_na = old_cloud
                y_ng = (corr/corr.sum(axis=1)[:,None]).dot(new_cloud)
                bend_coef = f._bend_coef
                rot_coef = f._rot_coef
                wt_n = f._wt_n.copy()
                
                interest_pts_inds = np.zeros(len(old_cloud), dtype=bool)
                if lr in closing_hmats:
                    interest_pts_inds += np.apply_along_axis(np.linalg.norm, 1, old_cloud - closing_hmats[lr][:3,3]) < 0.05
    
                interest_pts_err_tol = 0.0025
                max_iters = 5 if transferopt != "pose" else 0
                penalty_factor = 10.0
                
                if np.any(interest_pts_inds):
                    for _ in range(max_iters):
                        interest_pts_errs = np.apply_along_axis(np.linalg.norm, 1, (f.transform_points(x_na[interest_pts_inds,:]) - y_ng[interest_pts_inds,:]))
                        if np.all(interest_pts_errs < interest_pts_err_tol):
                            break
                        redprint("TPS fitting: The error of the interest points is above the tolerance. Increasing penalty for these weights.")
                        wt_n[interest_pts_inds] *= penalty_factor
                        GlobalVars.action_solvers[action].solve(wt_n, y_ng, bend_coef, rot_coef, f)
                        
        
                old_ee_traj = asarray(seg_info["%s_gripper_tool_frame"%lr]['hmat'][i_start - int(i_start > 0):i_end+1])
                transformed_ee_traj = f.transform_hmats(old_ee_traj)
                transformed_ee_traj_rs = np.asarray(resampling.interp_hmats(timesteps_rs, np.arange(len(transformed_ee_traj)), transformed_ee_traj))
                 
                if animate:
                    handles.append(sim_env.env.drawlinestrip(old_ee_traj[:,:3,3], 2, (1,0,0)))
                    handles.append(sim_env.env.drawlinestrip(transformed_ee_traj[:,:3,3], 2, (1,1,0)))
                    handles.append(sim_env.env.drawlinestrip(transformed_ee_traj_rs[:,:3,3], 2, (0,1,0)))
                    sim_env.viewer.Step()
                
                print "planning pose trajectory following"
                dof_inds = sim_util.dof_inds_from_name(sim_env.robot, manip_name)
                joint_ind = sim_env.robot.GetJointIndex("%s_shoulder_lift_joint"%lr)
                init_arm_traj = old_arm_traj_rs.copy()
                init_arm_traj[:,dof_inds.index(joint_ind)] = sim_env.robot.GetDOFLimits([joint_ind])[0][0]
                new_arm_traj, obj_value, pose_errs = planning.plan_follow_traj(sim_env.robot, manip_name, sim_env.robot.GetLink(ee_link_name), transformed_ee_traj_rs, init_arm_traj, 
                                                                               start_fixed=i_miniseg_lr!=0,
                                                                               use_collision_cost=False,
                                                                               beta_pos=beta_pos, beta_rot=beta_rot)
                
                if transferopt == 'finger' or transferopt == 'joint':
                    old_ee_traj_rs = np.asarray(resampling.interp_hmats(timesteps_rs, np.arange(len(old_ee_traj)), old_ee_traj))
                    old_finger_traj_rs = mu.interp2d(timesteps_rs, np.arange(len(old_finger_traj)), old_finger_traj)
                    flr2old_finger_pts_traj_rs = sim_util.get_finger_pts_traj(sim_env, lr, (old_ee_traj_rs, old_finger_traj_rs))
                    
                    flr2transformed_finger_pts_traj_rs = {}
                    flr2finger_link = {}
                    flr2finger_rel_pts = {}
                    for finger_lr in 'lr':
                        flr2transformed_finger_pts_traj_rs[finger_lr] = f.transform_points(np.concatenate(flr2old_finger_pts_traj_rs[finger_lr], axis=0)).reshape((-1,4,3))
                        flr2finger_link[finger_lr] = sim_env.robot.GetLink("%s_gripper_%s_finger_tip_link"%(lr,finger_lr))
                        flr2finger_rel_pts[finger_lr] = sim_util.get_finger_rel_pts(finger_lr)
                    
                    if animate:
                        handles.extend(sim_util.draw_finger_pts_traj(sim_env, flr2old_finger_pts_traj_rs, (1,0,0)))
                        handles.extend(sim_util.draw_finger_pts_traj(sim_env, flr2transformed_finger_pts_traj_rs, (0,1,0)))
                        sim_env.viewer.Step()
                        
                    # enable finger DOF and extend the trajectories to include the closing part only if the gripper closes at the end of this minisegment
                    next_gripper_open = sim_util.binarize_gripper(seg_info["%s_gripper_joint"%lr][i_end+1]) if i_end+1 < len(seg_info["%s_gripper_joint"%lr]) else True
                    if not sim_env.sim.is_grabbing_rope(lr) and not next_gripper_open:
                        manip_name = manip_name + "+" + "%s_gripper_l_finger_joint"%lr
                        
                        old_finger_closing_traj_start = old_finger_traj_rs[-1][0]
                        old_finger_closing_traj_target = sim_util.get_binary_gripper_angle(sim_util.binarize_gripper(seg_info["%s_gripper_joint"%lr][i_end+1]))
                        old_finger_closing_traj_rs = np.linspace(old_finger_closing_traj_start, old_finger_closing_traj_target, np.ceil(abs(old_finger_closing_traj_target - old_finger_closing_traj_start) / FINGER_CLOSE_RATE))[:,None]
                        closing_n_steps = len(old_finger_closing_traj_rs)
                        old_ee_closing_traj_rs = np.tile(old_ee_traj_rs[-1], (closing_n_steps,1,1))
                        flr2old_finger_pts_closing_traj_rs = sim_util.get_finger_pts_traj(sim_env, lr, (old_ee_closing_traj_rs, old_finger_closing_traj_rs))
                          
                        init_traj = np.r_[np.c_[new_arm_traj,                                   old_finger_traj_rs],
                                            np.c_[np.tile(new_arm_traj[-1], (closing_n_steps,1)), old_finger_closing_traj_rs]]
                        # init_traj = np.r_[np.c_[init_arm_traj,                                   old_finger_traj_rs],
                        #                     np.c_[np.tile(init_arm_traj[-1], (closing_n_steps,1)), old_finger_closing_traj_rs]]
                        flr2transformed_finger_pts_closing_traj_rs = {}
                        for finger_lr in 'lr':
                            flr2old_finger_pts_traj_rs[finger_lr] = np.r_[flr2old_finger_pts_traj_rs[finger_lr], flr2old_finger_pts_closing_traj_rs[finger_lr]]
                            flr2transformed_finger_pts_closing_traj_rs[finger_lr] = f.transform_points(np.concatenate(flr2old_finger_pts_closing_traj_rs[finger_lr], axis=0)).reshape((-1,4,3))
                            flr2transformed_finger_pts_traj_rs[finger_lr] = np.r_[flr2transformed_finger_pts_traj_rs[finger_lr],
                                                                                  flr2transformed_finger_pts_closing_traj_rs[finger_lr]]
                        
                        if animate:
                            handles.extend(sim_util.draw_finger_pts_traj(sim_env, flr2old_finger_pts_closing_traj_rs, (1,0,0)))
                            handles.extend(sim_util.draw_finger_pts_traj(sim_env, flr2transformed_finger_pts_closing_traj_rs, (0,1,0)))
                            sim_env.viewer.Step()
                    else:
                        init_traj = new_arm_traj
                        # init_traj = init_arm_traj
                    
                    new_traj, obj_value, pose_errs = planning.plan_follow_finger_pts_traj(sim_env.robot, manip_name, 
                                                                                          flr2finger_link, flr2finger_rel_pts, 
                                                                                          flr2transformed_finger_pts_traj_rs, init_traj, 
                                                                                          use_collision_cost=False,
                                                                                          start_fixed=i_miniseg_lr!=0,
                                                                                          beta_pos=beta_pos, gamma=gamma)

                    
                    if transferopt == 'joint':
                        print "planning joint TPS and finger points trajectory following"
                        new_traj, f, new_N_z, \
                        obj_value, rel_pts_costs, tps_cost = planning.joint_fit_tps_follow_finger_pts_traj(sim_env.robot, manip_name, flr2finger_link, flr2finger_rel_pts, flr2old_finger_pts_traj_rs, new_traj, 
                                                                                                           x_na, y_ng, bend_coef, rot_coef, wt_n, old_N_z=None,
                                                                                                           start_fixed=i_miniseg_lr!=0,
                                                                                                           alpha=alpha, beta_pos=beta_pos, gamma=gamma)
                        if np.any(interest_pts_inds):
                            for _ in range(max_iters):
                                interest_pts_errs = np.apply_along_axis(np.linalg.norm, 1, (f.transform_points(x_na[interest_pts_inds,:]) - y_ng[interest_pts_inds,:]))
                                if np.all(interest_pts_errs < interest_pts_err_tol):
                                    break
                                redprint("Joint TPS fitting: The error of the interest points is above the tolerance. Increasing penalty for these weights.")
                                wt_n[interest_pts_inds] *= penalty_factor
                                new_traj, f, new_N_z, \
                                obj_value, rel_pts_costs, tps_cost = planning.joint_fit_tps_follow_finger_pts_traj(sim_env.robot, manip_name, flr2finger_link, flr2finger_rel_pts, flr2old_finger_pts_traj_rs, new_traj, 
                                                                                                                   x_na, y_ng, bend_coef, rot_coef, wt_n, old_N_z=new_N_z,
                                                                                                                   start_fixed=i_miniseg_lr!=0,
                                                                                                                   alpha=alpha, beta_pos=beta_pos, gamma=gamma)
                    # else:
                    #     obj_value += alpha * planning.tps_obj(f, x_na, y_ng, bend_coef, rot_coef, wt_n)
                    
                    if animate:
                        flr2new_transformed_finger_pts_traj_rs = {}
                        for finger_lr in 'lr':
                            flr2new_transformed_finger_pts_traj_rs[finger_lr] = f.transform_points(np.concatenate(flr2old_finger_pts_traj_rs[finger_lr], axis=0)).reshape((-1,4,3))
                        handles.extend(sim_util.draw_finger_pts_traj(sim_env, flr2new_transformed_finger_pts_traj_rs, (0,1,1)))
                        sim_env.viewer.Step()
                else:
                    new_traj = new_arm_traj
                
                obj_values.append(obj_value)
                
                f._bend_coef = bend_coef
                f._rot_coef = rot_coef
                f._wt_n = wt_n
                
                full_traj = (new_traj, sim_util.dof_inds_from_name(sim_env.robot, manip_name))
                group_full_trajs.append(full_traj)
    
                if animate:
                    handles.append(sim_env.env.drawlinestrip(sim_util.get_ee_traj(sim_env, lr, full_traj)[:,:3,3], 2, (0,0,1)))
                    flr2new_finger_pts_traj = sim_util.get_finger_pts_traj(sim_env, lr, full_traj)
                    handles.extend(sim_util.draw_finger_pts_traj(sim_env, flr2new_finger_pts_traj, (0,0,1)))
                    sim_env.viewer.Step()
            full_traj = sim_util.merge_full_trajs(group_full_trajs)
        else:
            full_traj = replay_full_trajs[i_miniseg_group]
        full_trajs.append(full_traj)
        
        if not simulate:
            if not eval_util.traj_is_safe(sim_env, full_traj, COLLISION_DIST_THRESHOLD, upsample=100):
                return np.inf
            else:
                continue

        for (i_miniseg_lr, lr, _, _) in miniseg_interval_group:
            redprint("Executing %s arm joint trajectory for part %i"%(lr, i_miniseg_lr))
        
        if len(full_traj[0]) > 0:
            # if not eval_util.traj_is_safe(sim_env, full_traj, COLLISION_DIST_THRESHOLD, upsample=100):
            #     redprint("Trajectory not feasible")
            #     feasible = False
            #     success = False
            # else:  # Only execute feasible trajectories
            first_miniseg = True
            for (i_miniseg_lr, _, _, _) in miniseg_interval_group:
                first_miniseg &= i_miniseg_lr == 0
            if len(full_traj[0]) > 0:
                success &= sim_util.sim_full_traj_maybesim(sim_env, full_traj, animate=animate, interactive=interactive, max_cart_vel_trans_traj=.05 if first_miniseg else .02)

        if not success: break
        
        for (i_miniseg_lr, lr, i_start, i_end) in miniseg_interval_group:
            next_gripper_open = sim_util.binarize_gripper(seg_info["%s_gripper_joint"%lr][i_end+1]) if i_end+1 < len(seg_info["%s_gripper_joint"%lr]) else True
            curr_gripper_open = sim_util.binarize_gripper(seg_info["%s_gripper_joint"%lr][i_end])
            if not sim_util.set_gripper_maybesim(sim_env, lr, next_gripper_open, curr_gripper_open, animate=animate):
                redprint("Grab %s failed" % lr)
                misgrasp = True
                success = False

        if not success: break

    if not simulate:
        return np.sum(obj_values)

    sim_env.sim.settle(animate=animate)
    sim_env.sim.release_rope('l')
    sim_env.sim.release_rope('r')
    sim_util.reset_arms_to_side(sim_env)
    if animate:
        sim_env.viewer.Step()
    
    return success, feasible, misgrasp, full_trajs, get_state(sim_env, args_eval)

def get_unique_id(): 
    GlobalVars.unique_id += 1
    return GlobalVars.unique_id - 1

def get_state(sim_env, args_eval):
    if args_eval.raycast:
        new_cloud, endpoint_inds = sim_env.sim.raycast_cloud(endpoints=3)
        if new_cloud.shape[0] == 0: # rope is not visible (probably because it fall off the table)
            return None
    else:
        new_cloud = sim_env.sim.observe_cloud(upsample=args_eval.upsample, upsample_rad=args_eval.upsample_rad)
        endpoint_inds = np.zeros(len(new_cloud), dtype=bool) # for now, args_eval.raycast=False is not compatible with args_eval.use_color=True
    if args_eval.use_color:
        new_cloud = color_cloud(new_cloud, endpoint_inds)
    new_cloud_ds = clouds.downsample(new_cloud, DS_SIZE) if args_eval.downsample else new_cloud
    new_rope_nodes = sim_env.sim.rope.GetControlPoints()
    new_rope_nodes= ropesim.observe_cloud(new_rope_nodes, sim_env.sim.rope_params.radius, upsample=args_eval.upsample)
    init_rope_nodes = sim_env.sim.rope_pts
    rope_params = args_eval.rope_params
    tfs = sim_util.get_rope_transforms(sim_env)
    state = sim_util.RopeState("eval_%i"%get_unique_id(), new_cloud_ds, new_rope_nodes, init_rope_nodes, rope_params, tfs)
    return state
# @profile
def eval_on_holdout(args, sim_env):
    holdoutfile = h5py.File(args.eval.holdoutfile, 'r')
    holdout_items = eval_util.get_holdout_items(holdoutfile, args.tasks, args.taskfile, args.i_start, args.i_end)

    transfer = transfer_simulate.Transfer(args.eval, GlobalVars.action_solvers, register_tps) # signature of this class will change
    batch_transfer_simulate = transfer_simulate.BatchTransferSimulate(transfer, sim_env)

    num_successes = 0
    num_total = 0

    for i_task, demo_id_rope_nodes in holdout_items:
        redprint("task %s" % i_task)
        sim_util.reset_arms_to_side(sim_env)
        init_rope_nodes = demo_id_rope_nodes["rope_nodes"][:]
        sim_env.set_rope_state(init_rope_nodes, args.eval.rope_params)
        next_state = get_state(sim_env, args.eval)
        
        if args.animation:
            sim_env.viewer.Step()

        for i_step in range(args.eval.num_steps):
            redprint("task %s step %i" % (i_task, i_step))
            
            state = next_state

            num_actions_to_try = MAX_ACTIONS_TO_TRY if args.eval.search_until_feasible else 1
            eval_stats = eval_util.EvalStats()

            agenda, q_values_root = GlobalVars.features.select_best(state, num_actions_to_try)

            unable_to_generalize = False
            for i_choice in range(num_actions_to_try):
                if q_values_root[i_choice] == -np.inf: # none of the demonstrations generalize
                    unable_to_generalize = True
                    break
                redprint("TRYING %s"%agenda[i_choice])

                best_root_action = agenda[i_choice]
                start_time = time.time()
                # the following lines is the same as the last commented line
                batch_transfer_simulate.add_transfer_simulate_job(state, best_root_action, get_unique_id())
                results = batch_transfer_simulate.get_results()
                trajectory_result, next_state = results[0]
                eval_stats.success, eval_stats.feasible, eval_stats.misgrasp, full_trajs = \
                    trajectory_result.success, trajectory_result.feasible, trajectory_result.misgrasp, trajectory_result.full_trajs
                #eval_stats.success, eval_stats.feasible, eval_stats.misgrasp, full_trajs, next_state = compute_trans_traj(sim_env, state, best_root_action, args.eval, animate=args.animation, interactive=args.interactive)
                eval_stats.exec_elapsed_time += time.time() - start_time

                if eval_stats.feasible:  # try next action if TrajOpt cannot find feasible action
                     break
            if unable_to_generalize:
                 break
            print "BEST ACTION:", best_root_action

            if not eval_stats.feasible:  # If not feasible, restore state
                next_state = state
            
            eval_util.save_task_results_step(args.resultfile, i_task, i_step, state, best_root_action, q_values_root, full_trajs, next_state, eval_stats, new_cloud_ds=state.cloud, new_rope_nodes=state.rope_nodes)
            
            if not eval_stats.feasible:
                # Skip to next knot tie if the action is infeasible -- since
                # that means all future steps (up to 5) will have infeasible trajectories
                break
            
            if is_knot(next_state.rope_nodes):
                num_successes += 1
                break;

        num_total += 1

        redprint('Eval Successes / Total: ' + str(num_successes) + '/' + str(num_total))

def replay_on_holdout(args, sim_env):
    holdoutfile = h5py.File(args.eval.holdoutfile, 'r')
    loadresultfile = h5py.File(args.replay.loadresultfile, 'r')
    loadresult_items = eval_util.get_holdout_items(loadresultfile, args.tasks, args.taskfile, args.i_start, args.i_end)

    num_successes = 0
    num_total = 0
    
    for i_task, _ in loadresult_items:
        redprint("task %s" % i_task)

        for i_step in range(len(loadresultfile[i_task]) - (1 if 'init' in loadresultfile[i_task] else 0)):
            if args.replay.simulate_traj_steps is not None and i_step not in args.replay.simulate_traj_steps:
                continue
            
            redprint("task %s step %i" % (i_task, i_step))

            eval_stats = eval_util.EvalStats()

            state, best_action, q_values, replay_full_trajs, replay_next_state, _, _ = eval_util.load_task_results_step(args.replay.loadresultfile, i_task, i_step)

            unable_to_generalize = q_values.max() == -np.inf # none of the demonstrations generalize
            if unable_to_generalize:
                break
            
            start_time = time.time()
            if i_step in args.replay.compute_traj_steps: # compute the trajectory in this step
                replay_full_trajs = None            
            eval_stats.success, eval_stats.feasible, eval_stats.misgrasp, full_trajs, next_state = compute_trans_traj(sim_env, state, best_action, args.eval, animate=args.animation, interactive=args.interactive, replay_full_trajs=replay_full_trajs)
            eval_stats.exec_elapsed_time += time.time() - start_time
            print "BEST ACTION:", best_action

            if not eval_stats.feasible:  # If not feasible, restore state
                next_state = state
            
            if np.all(next_state.tfs[0] == replay_next_state.tfs[0]) and np.all(next_state.tfs[1] == replay_next_state.tfs[1]):
                yellowprint("Reproducible results OK")
            else:
                yellowprint("The rope transforms of the replay rope doesn't match the ones in the original result file by %f and %f" % (np.linalg.norm(next_state.tfs[0] - replay_next_state.tfs[0]), np.linalg.norm(next_state.tfs[1] - replay_next_state.tfs[1])))
            
            eval_util.save_task_results_step(args.resultfile, i_task, i_step, state, best_action, q_values, full_trajs, next_state, eval_stats)
            
            if not eval_stats.feasible:
                # Skip to next knot tie if the action is infeasible -- since
                # that means all future steps (up to 5) will have infeasible trajectories
                break
            
            if is_knot(next_state.rope_nodes):
                num_successes += 1
                break;

        num_total += 1

        redprint('REPLAY Successes / Total: ' + str(num_successes) + '/' + str(num_total))

def parse_input_args():
    parser = util.ArgumentParser()
    
    parser.add_argument("--animation", type=int, default=0, help="if greater than 1, the viewer tries to load the window and camera properties without idling at the beginning")
    parser.add_argument("--interactive", action="store_true", help="step animation and optimization if specified")
    parser.add_argument("--resultfile", type=str, help="no results are saved if this is not specified")

    # selects tasks to evaluate/replay
    parser.add_argument("--tasks", type=int, nargs='*', metavar="i_task")
    parser.add_argument("--taskfile", type=str)
    parser.add_argument("--i_start", type=int, default=-1, metavar="i_task")
    parser.add_argument("--i_end", type=int, default=-1, metavar="i_task")
    
    parser.add_argument("--camera_matrix_file", type=str, default='.camera_matrix.txt')
    parser.add_argument("--window_prop_file", type=str, default='.win_prop.txt')
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument("--log", type=str, default="")
    parser.add_argument("--print_mean_and_var", action="store_true")

    subparsers = parser.add_subparsers(dest='subparser_name')

    parser_eval = subparsers.add_parser('eval')
    
    parser_eval.add_argument('actionfile', type=str, nargs='?', default='data/misc/actions.h5')
    parser_eval.add_argument('holdoutfile', type=str, nargs='?', default='data/misc/holdout_set.h5')

    parser_eval.add_argument('--weightfile', type=str, default='')

    parser_eval.add_argument('warpingcost', type=str, choices=['regcost', 'regcost-trajopt', 'jointopt'])
    parser_eval.add_argument("transferopt", type=str, choices=['pose', 'finger', 'joint'])
    
    parser_eval.add_argument("--obstacles", type=str, nargs='*', choices=['bookshelve', 'boxes', 'cylinders'], default=[])
    parser_eval.add_argument("--raycast", type=int, default=0, help="use raycast or rope nodes observation model")
    parser_eval.add_argument("--downsample", type=int, default=1)
    parser_eval.add_argument("--upsample", type=int, default=0)
    parser_eval.add_argument("--upsample_rad", type=int, default=1, help="upsample_rad > 1 incompatible with downsample != 0")
    
    parser_eval.add_argument("--fake_data_segment",type=str, default='demo1-seg00')
    parser_eval.add_argument("--fake_data_transform", type=float, nargs=6, metavar=("tx","ty","tz","rx","ry","rz"),
        default=[0,0,0,0,0,0], help="translation=(tx,ty,tz), axis-angle rotation=(rx,ry,rz)")
    
    parser_eval.add_argument("--search_until_feasible", action="store_true")
    parser_eval.add_argument("--alpha", type=float, default=10000000.0)
    parser_eval.add_argument("--beta_pos", type=float, default=10000.0)
    parser_eval.add_argument("--beta_rot", type=float, default=10.0)
    parser_eval.add_argument("--gamma", type=float, default=1000.0)
    parser_eval.add_argument("--num_steps", type=int, default=5, help="maximum number of steps to simulate each task")
    parser_eval.add_argument("--use_color", type=int, default=0)
    parser_eval.add_argument("--dof_limits_factor", type=float, default=1.0)
    parser_eval.add_argument("--rope_params", type=str, default='default')

    parser_replay = subparsers.add_parser('replay')
    parser_replay.add_argument("loadresultfile", type=str)
    parser_replay.add_argument("--compute_traj_steps", type=int, default=[], nargs='*', metavar='i_step', help="recompute trajectories for the i_step of all tasks")
    parser_replay.add_argument("--simulate_traj_steps", type=int, default=None, nargs='*', metavar='i_step', 
                               help="if specified, restore the rope state from file and then simulate for the i_step of all tasks")
                               # if not specified, the rope state is not restored from file, but it is as given by the sequential simulation

    return parser.parse_args()

def get_scaled_action_cloud(state, action):
    ds_key = 'DS_SIZE_{}'.format(DS_SIZE)
    ds_g = GlobalVars.actions[action]['inv'][ds_key]
    scaled_x_na = ds_g['scaled_cloud_xyz'][:]
    src_params = (ds_g['scaling'][()], ds_g['scaled_translation'][:])
    return scaled_x_na, src_params

def get_action_cloud(sim_env, action, args_eval):
    rope_nodes = get_action_rope_nodes(sim_env, action, args_eval)
    cloud = ropesim.observe_cloud(rope_nodes, ROPE_RADIUS, upsample_rad=args_eval.upsample_rad)
    return cloud

def get_action_cloud_ds(sim_env, action, args_eval):
    if args_eval.downsample:
        ds_key = 'DS_SIZE_{}'.format(DS_SIZE)
        return GlobalVars.actions[action]['inv'][ds_key]['cloud_xyz']
    else:
        return get_action_cloud(sim_env, action, args_eval)

def get_action_rope_nodes(sim_env, action, args_eval):
    rope_nodes = GlobalVars.actions[action]['cloud_xyz'][()]
    return ropesim.observe_cloud(rope_nodes, ROPE_RADIUS, upsample=args_eval.upsample)


def setup_log_file(args):
    if args.log:
        redprint("Writing log to file %s" % args.log)
        GlobalVars.exec_log = task_execution.ExecutionLog(args.log)
        atexit.register(GlobalVars.exec_log.close)
        GlobalVars.exec_log(0, "main.args", args)

def set_global_vars(args):
    if args.random_seed is not None: np.random.seed(args.random_seed)
    GlobalVars.actions = h5py.File(args.eval.actionfile, 'r')
    actions_root, actions_ext = os.path.splitext(args.eval.actionfile)
    GlobalVars.actions_cache = h5py.File(actions_root + '.cache' + actions_ext, 'a')
    GlobalVars.features = BatchRCFeats(args.eval.actionfile)
    if args.eval.weightfile:
        GlobalVars.features.load_weights(args.eval.weightfile)
    GlobalVars.action_solvers = TPSSolver.get_solvers(GlobalVars.actions)
    exact_bend_coefs = np.around(loglinspace(EXACT_LAMBDA[0], EXACT_LAMBDA[1], N_ITER_EXACT), BEND_COEF_DIGITS)
    GlobalVars.empty_solver = EmptySolver(MAX_CLD_SIZE, exact_bend_coefs)

def load_simulation(args):
    actions = h5py.File(args.eval.actionfile, 'r')
    
    init_rope_xyz, init_joint_names, init_joint_values = sim_util.load_fake_data_segment(actions, args.eval.fake_data_segment, args.eval.fake_data_transform) 
    table_height = init_rope_xyz[:,2].mean() - .02

    sim_env = sim_util.SimulationEnv(table_height, init_joint_names, init_joint_values, args.eval.obstacles, args.eval.dof_limits_factor)
    sim_env.initialize()
    
    if args.animation:
        sim_env.viewer = trajoptpy.GetViewer(sim_env.env)
        if args.animation > 1 and os.path.isfile(args.window_prop_file) and os.path.isfile(args.camera_matrix_file):
            print "loading window and camera properties"
            window_prop = np.loadtxt(args.window_prop_file)
            camera_matrix = np.loadtxt(args.camera_matrix_file)
            try:
                sim_env.viewer.SetWindowProp(*window_prop)
                sim_env.viewer.SetCameraManipulatorMatrix(camera_matrix)
            except:
                print "SetWindowProp and SetCameraManipulatorMatrix are not defined. Pull and recompile Trajopt."
        else:
            print "move viewer to viewpoint that isn't stupid"
            print "then hit 'p' to continue"
            sim_env.viewer.Idle()
            print "saving window and camera properties"
            try:
                window_prop = sim_env.viewer.GetWindowProp()
                camera_matrix = sim_env.viewer.GetCameraManipulatorMatrix()
                np.savetxt(args.window_prop_file, window_prop, fmt='%d')
                np.savetxt(args.camera_matrix_file, camera_matrix)
            except:
                print "GetWindowProp and GetCameraManipulatorMatrix are not defined. Pull and recompile Trajopt."
    
    return sim_env

def main():
    args = parse_input_args()

    if args.subparser_name == "eval":
        eval_util.save_results_args(args.resultfile, args)
    elif args.subparser_name == "replay":
        loaded_args = eval_util.load_results_args(args.replay.loadresultfile)
        assert 'eval' not in vars(args)
        args.eval = loaded_args.eval
    else:
        raise RuntimeError("Invalid subparser name")
    
    setup_log_file(args)
    
    set_global_vars(args)
    trajoptpy.SetInteractive(args.interactive)
    sim_env = load_simulation(args)

    if args.subparser_name == "eval":
        start = time.time()
        eval_on_holdout(args, sim_env)
        print "eval time is:\t{}".format(time.time() - start)
    elif args.subparser_name == "replay":
        replay_on_holdout(args, sim_env)
    else:
        raise RuntimeError("Invalid subparser name")

    if args.print_mean_and_var:
        if GlobalVars.tps_errors_top10:
            print "TPS error mean:", np.mean(GlobalVars.tps_errors_top10)
            print "TPS error variance:", np.var(GlobalVars.tps_errors_top10)
            print "Total Num TPS errors:", len(GlobalVars.tps_errors_top10)
        if GlobalVars.trajopt_errors_top10:
            print "TrajOpt error mean:", np.mean(GlobalVars.trajopt_errors_top10)
            print "TrajOpt error variance:", np.var(GlobalVars.trajopt_errors_top10)
            print "Total Num TrajOpt errors:", len(GlobalVars.trajopt_errors_top10)

if __name__ == "__main__":
    main()
