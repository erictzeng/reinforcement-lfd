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

from tpsopt.registration import tps_rpm_bij, loglinspace
from tpsopt.transformations import TPSSolver, EmptySolver

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

    transfer = transfer_simulate.Transfer(args.eval, register_tps) # signature of this class will change
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
                results = batch_transfer_simulate.get_results(animate=args.animation)
                trajectory_result, next_state, next_state_id = results[0]
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
    parser.add_argument("--landmarkfile", type=str, default='../data/misc/landmarks.h5')

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
    parser_eval.add_argument('feature_type', type=str, nargs='?', choices=['base', 'mul', 'mul_quad', 'mul_s', 'landmark'], default='base')

    parser_eval.add_argument('warpingcost', type=str, nargs='?', choices=['regcost', 'regcost-trajopt', 'jointopt'], default='regcost')
    parser_eval.add_argument("transferopt", type=str, nargs='?', choices=['pose', 'finger', 'joint'], default='finger')
    
    
    parser_eval.add_argument("--obstacles", type=str, nargs='*', choices=['bookshelve', 'boxes', 'cylinders'], default=[])
    parser_eval.add_argument("--raycast", type=int, default=0, help="use raycast or rope nodes observation model")
    parser_eval.add_argument("--downsample", type=int, default=1)
    parser_eval.add_argument("--upsample", type=int, default=0)
    parser_eval.add_argument("--upsample_rad", type=int, default=1, help="upsample_rad > 1 incompatible with downsample != 0")
    
    parser_eval.add_argument("--fake_data_segment",type=str, default='demo1-seg00')
    parser_eval.add_argument("--fake_data_transform", type=float, nargs=6, metavar=("tx","ty","tz","rx","ry","rz"),
        default=[0,0,0,0,0,0], help="translation=(tx,ty,tz), axis-angle rotation=(rx,ry,rz)")
    
    parser_eval.add_argument("--search_until_feasible", action="store_true")
    parser_eval.add_argument("--alpha", type=float, default=1000000.0)
    parser_eval.add_argument("--beta_pos", type=float, default=1000000.0)
    parser_eval.add_argument("--beta_rot", type=float, default=1.0)
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

def get_features(args):
    feat_type = args.eval.feature_type
    if feat_type == 'base':
        from features import BatchRCFeats as feat
    elif feat_type == 'mul':
        from features import MulFeats as feat
    elif feat_type == 'mul_quad':
        from features import QuadMulFeats as feat
    elif feat_type == 'mul_s':
        from features import SimpleMulFeats as feat
    elif feat_type == 'landmark':
        from features import LandmarkFeats as feat
    else:
        raise ValueError('Incorrect Feature Type')
    
    feats = feat(args.eval.actionfile)
    try:
        feats.set_landmark_file(args.landmarkfile)
    except AttributeError:
        pass
    return feats



def set_global_vars(args):
    if args.random_seed is not None: np.random.seed(args.random_seed)
    GlobalVars.actions = h5py.File(args.eval.actionfile, 'r')
    actions_root, actions_ext = os.path.splitext(args.eval.actionfile)
    GlobalVars.actions_cache = h5py.File(actions_root + '.cache' + actions_ext, 'a')
    GlobalVars.features = get_features(args)
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
