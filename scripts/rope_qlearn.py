#!/usr/bin/env python

"""
code for doing max margin Qlearning for knot-tying
includes helper functions for manipulating demonstration
and action data
"""

import argparse

import h5py, math, random
import IPython as ipy
from max_margin import MaxMarginModel, MultiSlackMaxMarginModel, BellmanMaxMarginModel
from pdb import pm, set_trace
from knot_classifier import isKnot as is_knot
import traj_utils
import numpy as np
from joblib import Parallel, delayed
import scipy.spatial.distance as ssd
import scipy.spatial as sp_spat
import os.path
import cProfile
import util as u
import sys
from rapprentice import registration, tps_registration_parallel
from rapprentice.registration import ThinPlateSpline # needs to be defined in order to be defined properly in the cluster

DS_SIZE = .025
GRIPPER_OPEN_CLOSE_THRESH = 0.04

# constants for shape context
R_MIN, P_MIN, T_MIN = 0, 0, 0
R_MAX, P_MAX, T_MAX = 0.15, np.pi, 2*np.pi
R_BINS, P_BINS, T_BINS = 4, 1, 4
DENSITY_RADIUS = .2

DONE_MARGIN_VALUE = 175


# rip rope_max_margin_model, you will not be missed --eric 1/14/2014

def compute_constraints_no_model(feature_fn, margin_fn, act_set, expert_demofile, outfile, start=0, end=-1, verbose=False):
    """
    computes all the constraints associated with expert_demofile, output is consistent with files saved
    from max_margin.MultiSlackMaxMarginModel

    still needs a little testing
    """
    if type(expert_demofile) is str:
        expert_demofile = h5py.File(expert_demofile, 'r')
    if type(outfile) is str:
        outfile = h5py.File(outfile, 'w')
    if verbose:
        print "adding constraints"
    constraint_ctr = 0
    if end < 0:
        end = len(expert_demofile)
    for demo_i in range(start, end):
        key = str(demo_i)
        group = expert_demofile[key]
        state = [key,group['cloud_xyz'][:]] # these are already downsampled
        action = group['action'][()]
        if action.startswith('endstate'): # this is a knot
            continue
        if 'orig_action' in group.keys():
            orig_action = group['orig_action'][()]
        else:
            orig_action = ""
        if verbose:
            print 'adding constraints for:\t', action        
        lhs_phi = feature_fn(state, action)
        xi_name = str('xi_') + str(key)
        for (i, other_a) in enumerate(act_set.actions):
            if other_a == action or other_a == orig_action:
                continue
            if verbose:
                print "added {}/{}".format(i, len(act_set.actions))
            rhs_phi = feature_fn(state, other_a)
            margin = margin_fn(state, action, other_a)
            g = outfile.create_group(str(constraint_ctr))
            constraint_ctr += 1
            g['example'] = key
            g['exp_features'] = lhs_phi
            g['rhs_phi'] = rhs_phi
            g['margin'] = margin
            g['xi'] = xi_name
        outfile.flush()
    outfile.close()

def compute_bellman_constraints_no_model(feature_fn, margin_fn, act_set, expert_demofile, outfile, start=0, end=-1, verbose=False, parallel=False, ppservers=(), ignore_bellman_constraints=False):
    if type(expert_demofile) is str:
        expert_demofile = h5py.File(expert_demofile, 'r')
    if type(outfile) is str:
        outfile = h5py.File(outfile, 'w')
    if verbose:
        print "adding constraints"
    
    if end < 0:
        end = len(expert_demofile)
    while start < len(expert_demofile) and int(expert_demofile[str(start)]['pred'][()]) != start:
        start += 1
    while end < len(expert_demofile) and int(expert_demofile[str(end)]['pred'][()]) != end:
        end += 1
    
    if parallel:
        states = []
        done_states = []
        for demo_i in range(start, end):
            key = str(demo_i)
            group = expert_demofile[key]
            state = [key,group['cloud_xyz'][:]] # these are already downsampled
            action = group['action'][()]
            action = action if not group['knot'][()] else 'done'
            if action != 'done':
                states.append(state)
            else:
                done_states.append(state) # need feature of these for bellman constraints
        registration_cost_cheap_parallel_precompute(act_set, states, act_set.actions, ppservers=ppservers)
        warp_hmats_parallel_precompute(act_set, states, act_set.actions, ppservers=ppservers)
        if ignore_bellman_constraints:
            landmark_features_parallel_precompute(act_set, states, ppservers=ppservers)
        else:
            landmark_features_parallel_precompute(act_set, states+done_states, ppservers=ppservers)

    trajectories = []
    traj = []
    constraint_ctr = 0
    for demo_i in range(start, end):
        key = str(demo_i)
        group = expert_demofile[key]
        state = [key,group['cloud_xyz'][:]] # these are already downsampled
        action = group['action'][()]
        action = action if not group['knot'][()] else 'done'
        # add examples
        if action != 'done':
            lhs_phi = feature_fn(state, action)
            xi_name = str('xi_') + str(key)
            for (i, other_a) in enumerate(act_set.actions):
                if other_a == action:
                    continue
                if verbose:
                    sys.stdout.write("added {}/{} for max_margin constraint {}\r".format(i, len(act_set.actions), xi_name))
                    sys.stdout.flush()
                rhs_phi = feature_fn(state, other_a)
                margin = margin_fn(state, action, other_a)
                g = outfile.create_group(str(constraint_ctr))
                constraint_ctr += 1
                g['example'] = key
                g['action'] = action
                g['exp_features'] = lhs_phi
                g['rhs_phi'] = rhs_phi
                g['margin'] = margin
                g['xi'] = xi_name
        # trajectories for bellman_constraints
        if group['pred'][()] == key:
            if traj:
                trajectories.append(traj)
                traj = []
        traj.append([state, action])
        outfile.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()
    if not ignore_bellman_constraints:
        if traj:
            trajectories.append(traj)
            traj = []
        for traj in trajectories:
            # add bellman constraints
            yi_name = 'yi_%s'%traj[0][0][0] # use the state id of the first trajectory as the trajectory id
            for i in range(len(traj)-1):
                curr_state, curr_action = traj[i]
                next_state, next_action = traj[i+1]
                lhs_action_phi = feature_fn(curr_state, curr_action)
                rhs_action_phi = feature_fn(next_state, next_action)
                g = outfile.create_group(str(constraint_ctr))
                constraint_ctr += 1
                g['example'] = '{}_{}_step{}'.format(curr_state[0], next_state[0], i)
                g['curr_action'] = curr_action
                g['next_action'] = next_action
                g['exp_features'] = lhs_action_phi
                g['rhs_phi'] = rhs_action_phi
                g['margin'] = 0
                g['xi'] = yi_name
                if verbose:
                    sys.stdout.write("added bellman constraint {}/{}\r ".format(i, len(traj)-1) + str(yi_name))
                    sys.stdout.flush()
            sys.stdout.write('\n')
            outfile.flush()
    outfile.close()

def add_constraints_from_demo(mm_model, expert_demofile, start=0, end=-1, outfile=None, verbose=False):
    """
    takes all of the expert demonstrations from expert_demofile
    and add all of the associated constrainted to mm_model

    if outfile is specified, will store the constraints in that file

    probably want to run once, save results and then specify constraints to 
    rope_max_margin_model so we don't have to recompute features
    """
    if type(expert_demofile) is str:
        expert_demofile = h5py.File(expert_demofile, 'r')
    if verbose:
        print "adding constraints"
    
    if end < 0:
        end = len(expert_demofile)
    while int(expert_demofile[str(start)]['pred'][()]) != start:
        start += 1
    while end < len(expert_demofile)-1 and int(expert_demofile[str(end)]['pred'][()]) != end:
        end += 1

    for demo_i in range(start, end):  
        # Assumes example ids are strings of consecutive integers starting from 0
        key = str(demo_i)
        group = expert_demofile[key]
        state = [key,group['cloud_xyz'][:]] # these are already downsampled
        action = group['action'][()]
        if action.startswith('endstate'): # this is a knot
            continue
        if 'orig_action' in group.keys():
            orig_action = group['orig_action'][()]
        else:
            orig_action = ""
        if verbose:
            print 'adding constraints for:\t', action
        xi_name = str('xi_') + str(key)
        mm_model.add_example(state, action, xi_name=xi_name, verbose=verbose, orig_action=orig_action)
        #mm_model.clear_asm_cache()
        if outfile:
            mm_model.save_constraints_to_file(outfile)

def add_bellman_constraints_from_demo(mm_model, expert_demofile, start=0, end=-1, outfile=None, verbose=False):
    if type(expert_demofile) is str:
        expert_demofile = h5py.File(expert_demofile, 'r')
    if verbose:
        print "adding constraints"

    if end < 0:
        end = len(expert_demofile)
    while start < len(expert_demofile) and int(expert_demofile[str(start)]['pred'][()]) != start:
        start += 1
    while end < len(expert_demofile) and int(expert_demofile[str(end)]['pred'][()]) != end:
        end += 1

    trajectories = []
    traj = []
    for demo_i in range(start, end):
        key = str(demo_i)
        group = expert_demofile[key]
        state = [key,group['cloud_xyz'][:]] # these are already downsampled
        action = group['action'][()]
        if action.startswith('endstate'): # this is a knot
            continue
        if verbose:
            print 'adding constraints for:\t', action
        xi_name = str('xi_') + str(key)
        mm_model.add_example(state, action, xi_name=xi_name, verbose=verbose)
        if group['pred'][()] == key:
            if traj:
                trajectories.append(traj)
                traj = []
        traj.append([state, action])
        if outfile:
            mm_model.save_constraints_to_file(outfile)
    if traj:
        trajectories.append(traj)
        traj = []
    for traj in trajectories:
        if verbose:
            print "adding trajectory for trajectory with actions:\n", [a for [s,a] in traj]
        yi_name = 'yi_%s'%traj[0][0][0] # use the state id of the first trajectory as the trajectory id
        mm_model.add_trajectory(traj, yi_name, verbose)
        if outfile:
            mm_model.save_constraints_to_file(outfile)

def concatenate_fns(fns, act_set):
    fn_params = [f(act_set) for f in fns]
    def result_fn(state, action):
        results = np.array([])
        for (f, _) in fn_params:
            results = np.r_[results, f(state, action)]
        return results

    N = sum(v[1] for v in fn_params)
    return (result_fn, N)

def apply_rbf(ft_fn):
    def new_ft_fn(state, action):
        ft = ft_fn(state, action)
        new_ft = np.exp(-np.square(ft))
        new_ft /= np.linalg.norm(new_ft, 1)
        return new_ft
    return new_ft_fn

def get_traj_diff_feature_fn(act_set):
    return (act_set.traj_diff_features, act_set.num_traj_diff_features)

def get_is_knot_feature_fn(act_set):
    return (act_set.is_knot_features, act_set.num_is_knot_features)

def get_done_feature_fn(act_set):
    return (act_set.done_features, act_set.num_done_features)

def get_landmark_feature_fn(act_set, rbf=False):
    ft_fn = act_set.landmark_features
    if rbf:
        print 'Applying RBF to landmark features.'
        ft_fn = apply_rbf(ft_fn)
    return (ft_fn, act_set.num_landmark_features)

def get_sc_feature_fn(act_set):
    return (act_set.sc_features, act_set.num_sc_features)

def get_rope_dist_feat_fn(act_set):
    return (act_set.rope_dist_features, act_set.num_rope_dist_feat)

def get_bias_feature_fn(act_set):
    def feature_fn(state, action):
        return act_set.bias_features(state, action)
    return (feature_fn, act_set.num_actions + 1)

def get_quad_feature_fn(act_set):
    def feature_fn(state, action):
        return act_set.quad_features(state, action)
    return (feature_fn, 2 + 2*act_set.num_actions)

def get_quad_feature_noregcostsq_fn(act_set):
    def feature_fn(state, action):
        return act_set.quad_features_noregcostsq(state, action)
    return (feature_fn, 1 + 2*act_set.num_actions)

def get_action_only_margin_fn(act_set):
    return act_set.action_only_margin

def get_action_state_margin_fn(act_set):
    return act_set.action_state_margin

class ActionSet(object):
    """
    class to handle computing features and margins for state/action pairs


    state is assumed to be a list [<state_id>, <point_cloud>]
    """
    def __init__(self, actionfile, landmarks=[], gripper_weighting = False, use_cache = True, downsample = True):
        # set up openrave env for traj cost
        self.env, self.robot = traj_utils.initialize_lite_sim()
        
        if type(actionfile) is str:
            self.actionfile = h5py.File(actionfile, 'r')
        else:
            self.actionfile = actionfile
        self.actions = sorted(self.actionfile.keys())
        self.actions_ds_clouds = {}
        if downsample:
            from rapprentice import clouds
            for action in self.actions:
                self.actions_ds_clouds[action] = clouds.downsample(self.actionfile[action]['cloud_xyz'], DS_SIZE)
        else:
            for action in self.actions:
                self.actions_ds_clouds[action] = self.actionfile[action]['cloud_xyz'][()]
        # not including 'done' as an action anymore in max-margin constraints
        #self.actions.append('done')
        self.action_to_ind = dict((v, i) for i, v in enumerate(self.actions))
        self.num_actions = len(self.actions)
        self.num_is_knot_features = 1
        self.num_done_features = 2
        self.num_sc_features = R_BINS*T_BINS*P_BINS*2
        self.num_rope_dist_feat = 3
        self.num_traj_diff_features = 1
        self.gripper_weighting = gripper_weighting
        self.use_cache = use_cache
        if use_cache:
            self.caches = {}
            self.caches['warp_hmats'] = {}
            self.caches['landmarks'] = {}
            self.caches['cheap_reg_costs'] = {}
        self.link_names = ["%s_gripper_tool_frame"%lr for lr in ('lr')]
        if type(landmarks) is str:
            self.landmarks = h5py.File(landmarks, 'r')
        else:
            self.landmarks = landmarks
        self.landmark_knot_indices = [int(i) for i in self.landmarks if self.landmarks[i]['knot'][()]]
        self.num_landmark_features = len(self.landmarks)

    def _warp_hmats(self, state, action):
        key = (state[0], action)
        if self.use_cache and key in self.caches['warp_hmats']:
            return self.caches['warp_hmats'][key]
        if self.gripper_weighting:
            raise NotImplementedError, "Cache for warp_hmats don't keep track of this alternative version"
        [warped_trajs, rc, warped_rope_xyz] = warp_hmats(self.actions_ds_clouds[action],
                                              state[1],
                                              [(lr, self.actionfile[action][ln]['hmat']) for lr, ln in zip('lr', self.link_names)])
        if self.use_cache:
            self.caches['warp_hmats'][key] = [warped_trajs, rc, warped_rope_xyz]
        return [warped_trajs, rc, warped_rope_xyz]

    def _registration_cost_cheap(self, state, action):
        key = (state[0], action)
        if self.use_cache and key in self.caches['cheap_reg_costs']:
            return self.caches['cheap_reg_costs'][key]
        reg_cost = registration_cost_cheap(self.actions_ds_clouds[action], state[1])
        if self.use_cache:
            self.caches['cheap_reg_costs'][key] = reg_cost
        return reg_cost

    def traj_diff_features(self, state, action):
        if action == 'done':
            return np.array([0])
        target_trajs = self._warp_hmats(state, action)[0]
        orig_joint_trajs = traj_utils.joint_trajs(action, self.actionfile)
        err = traj_utils.follow_trajectory_cost(target_trajs, orig_joint_trajs, self.robot)
        return np.array([err])

    def is_knot_features(self, state, action):
        return np.array([int(is_knot(state[1]))])

    def done_features(self, state, action):
        if action != 'done':
            return np.array([0, 0])
        else:
            lm = self.landmark_features(state, action)
            regcost = min(lm[self.landmark_knot_indices])
            return np.array([1, regcost])
    
    def landmark_features(self, state, action):
        if self.use_cache and state[0] in self.caches['landmarks']:
            return self.caches['landmarks'][state[0]]
        feat = np.empty(len(self.landmarks))
        for i in range(len(self.landmarks)):
            landmark = self.landmarks[str(i)]
            feat[i] = registration_cost_cheap(landmark['cloud_xyz'][()], state[1])
        if self.use_cache:
            self.caches['landmarks'][state[0]] = feat
        return feat

    def sc_features(self, state, action):
        if action == 'done':
            return np.zeros(self.num_sc_features)

        seg_info = self.actionfile[action]
        warped_trajs, _, _ = self._warp_hmats(state, action)
        feat_val = dict((lr, np.zeros(self.num_sc_features/2.0)) for lr in 'lr')
        closings = get_closing_inds(seg_info)
        for lr in 'lr':
            first_close = closings[lr]
            if first_close != -1:
                close_hmat = warped_trajs[lr][first_close]
                feat_val[lr] = gripper_frame_shape_context(state[1], close_hmat)
                feat_norm = np.linalg.norm(feat_val[lr], ord=2)
                if feat_norm > 0:
                    feat_val[lr] = feat_val[lr] / feat_norm
        return np.r_[feat_val['l'], feat_val['r']]            

    def rope_dist_features(self, state, action):
        """
        Feature 1: sum of distances from each point in new rope to closest
        point in warped original rope, weighted by the distance of the new
        rope point to the closest gripper trajectory point
        Feature 2: distance of closest new rope point to left gripper, at
        its position right before closing
        Feature 3: same as Feature 2, but for the right gripper
        """
        feat = np.zeros(self.num_rope_dist_feat)
        if action == 'done':
            return feat
        new_rope_xyz = state[1]
        _, _, warped_rope_xyz = self._warp_hmats(state, action)
        kd_warped_rope = sp_spat.KDTree(warped_rope_xyz)
        weights = compute_weights(new_rope_xyz, get_traj_pts(self.actionfile[action]))
        dist_btwn_ropes = kd_warped_rope.query(new_rope_xyz)[0]
        feat[0] = np.dot(dist_btwn_ropes, weights)
        gripper_closing_pts = get_closing_pts(self.actionfile[action], as_dict = True)
        for ind, lr in enumerate('lr'):
            if lr in gripper_closing_pts:
                feat[1 + ind] = kd_warped_rope.query(gripper_closing_pts[lr])[0]
        return feat
    
    def bias_features(self, state, action):
        feat = np.zeros(self.num_actions + 1)
        if action == 'done':
            return feat
        feat[0] = self._registration_cost_cheap(state, action)
        feat[self.action_to_ind[action]+1] = 1
        return feat
    
    def quad_features(self, state, action):
        feat = np.zeros(2 + 2*self.num_actions)
        if action == 'done':
            return feat
        s = self._registration_cost_cheap(state, action)
        feat[0] = s**2
        feat[1] = s
        feat[2+self.action_to_ind[action]] = s
        feat[2+self.num_actions+self.action_to_ind[action]] = 1
        return feat

    def quad_features_noregcostsq(self, state, action):
        feat = np.zeros(1 + 2*self.num_actions)
        if action == 'done':
            return feat
        s = self._registration_cost_cheap(state, action)
        feat[0] = s
        feat[1+self.action_to_ind[action]] = s
        feat[1+self.num_actions+self.action_to_ind[action]] = 1
        return feat

    def action_only_margin(self, s, a1, a2):
        """
        warp both actions, compare the resulting trajectories:
        ex. warp a1 -> a2; use compare_hmats(warp(a1.traj), a2.traj)
        """
        # Removed 'done' from set of possible actions
        assert 'done' not in (a1, a2), "There is no 'done' action"
        #if 'done' in (a1, a2):
        #    return DONE_MARGIN_VALUE
        warped_a1_trajs, _, _ = self._warp_hmats((a2, self.get_ds_cloud(a2)), a1)
        warped_a1_trajs = [warped_a1_trajs[lr] for lr in 'lr']
        a1_trajs = [self.actionfile[a1][ln]['hmat'][:] for ln in self.link_names]
        warped_a2_trajs, _, _ = self._warp_hmats((a1, self.get_ds_cloud(a1)), a2)
        warped_a2_trajs = [warped_a2_trajs[lr] for lr in 'lr']
        a2_trajs = [self.actionfile[a2][ln]['hmat'][:] for ln in self.link_names]
        ret_val = sum(compare_hmats(t1, t2) for (t1, t2) in 
                          zip(warped_a1_trajs + warped_a2_trajs, a2_trajs + a1_trajs))
        return ret_val

    def check_cache(self, s, a):
        # cache stores the warped traj for a
        if self.use_cache:
            key = (s[0], a)
            if key in self.cache:
                return (True, self.cache[key])
        return False, False

    def store_cache(self, s, a, v):
        if self.use_cache:
            key = (s[0], a)
            self.cache[key] = v

    def clear_cache(self):
        self.cache = {}

    def action_state_margin(self, state, a1, a2):
        """
        look at the difference for both when warped to state
        ex. w1 is warp(a1, state), w2 is warp(a2, state)
        compare_hmats(w1(a1.traj), w2(a2.traj))

        might be worth it to implement cacheing at state/action level here
        when we call this with a particular expert demo we will warp that trajectory
        once for each action we compare to -- issue is hashing point clouds effectively        
        """
        # Removed 'done' from set of possible actions
        assert 'done' not in (a1, a2), "There is no 'done' action"
        #if 'done' in (a1, a2):
        #    return DONE_MARGIN_VALUE
        warped_a1_trajs, _, _ = self._warp_hmats(state, a1)
        warped_a2_trajs, _, _ = self._warp_hmats(state, a2)
        return sum(compare_hmats(warped_a1_trajs[lr], warped_a2_trajs[lr]) for lr in 'lr')

    def combined_margin(self, state, a1, a2):
        return (self.action_only_margin(a1, a2) +
                self.action_state_margin(a1, a2, state))

#http://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
# returns (r, theta, phi) with positive r, theta in [0, 2*pi], and phi in [0, pi]
def cart2spherical(xyz):
    xyz = np.asarray(xyz)
    ptsnew =  np.zeros(xyz.shape)
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,0] = np.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,1] = np.arctan2(xyz[:,1], xyz[:,0])
    ptsnew[ptsnew[:,1] < 0, 1] += 2*np.pi
    ptsnew[:,2] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,2] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    return ptsnew

def get_sect_vol((r1, r2), (t1, t2), (p1, p2)):
    return float(r2**3 - r1**3)/3.0 * (t2 - t1) * (-math.cos(p2) + math.cos(p1))

def gripper_frame_shape_context(xyz, hmat):
    # Project gripper center onto the plane of the table --> then this point is
    # the center of the shape context feature we compute
    h_inv = np.linalg.inv(hmat)
    table_height = xyz[:,2].mean() - 0.02
    fwd_axis = np.array([1, 0, 0, 1], 'float')  # homogeneous coord for e1
                                                # (in coord system of PR2, x-axis is forward)
    # Rotate forward axis to the frame of the gripper
    fwd_axis_gripper = np.dot(h_inv, fwd_axis)[:3]
    gripper_trans = hmat[:3, 3]

    # calculation of projection of gripper center, gripper_proj:
    # gripper_proj = gripper_trans + c * fwd_axis_gripper,
    # where c is chosen so that gripper_proj is at the same elevation (z-axis) as the table
    c = float(table_height - gripper_trans[2]) / fwd_axis_gripper[2]
    gripper_proj = gripper_trans + c * fwd_axis_gripper

    hmat_table = np.copy(hmat)
    gripper_trans_homog = np.ones((1, 4), 'float')
    gripper_trans_homog[0, :3] = gripper_proj
    hmat_table[:,3] = gripper_trans_homog
    h_inv_table = np.linalg.inv(hmat_table)

    xyz1 = np.ones((len(xyz),4),'float')  #homogeneous coord
    xyz1[:,:3] = xyz
    xyz2 = [np.dot(h_inv_table, pt)[:3] for pt in xyz1] #bestpractices
    rot_axes = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    # Rotate so z-axis becomes y-axis, y becomes x, and x becomes z.
    # Then theta is the angle we are interested in (since yz plane approximately
    # corresponds to the plane of the table)
    xyz2_rot = [np.dot(rot_axes, pt) for pt in xyz2]
    xyz3 = cart2spherical(xyz2_rot)

    bin_volumes = {}
    bin_weights = {}

    kd = sp_spat.KDTree(xyz)

    rvals = np.linspace(R_MIN, R_MAX, R_BINS+1)
    tvals = np.linspace(T_MIN, T_MAX, T_BINS+1)
    pvals = np.linspace(P_MIN, P_MAX, P_BINS+1)

    rbin_width = rvals[1] - rvals[0]
    tbin_width = tvals[1] - tvals[0]
    pbin_width = pvals[1] - pvals[0]

    for (pt_i, (rho, theta, phi)) in enumerate(xyz3):
        # Get point density: # of points in a sphere of radius DENSITY_RADIUS around this point
        pt_density = len(kd.query_ball_point(xyz[pt_i], DENSITY_RADIUS))

        # Find bin indices for point - correspond to the lower limits of the
        # bins that the point is in
        if rho > R_MAX or rho < R_MIN or theta > T_MAX or theta < T_MIN \
                or phi > P_MAX or phi < P_MIN: continue
        r_ind = math.floor((rho-rvals[0]) / rbin_width)
        t_ind = math.floor((theta-tvals[0]) / tbin_width)
        p_ind = math.floor((phi-pvals[0]) / pbin_width)

        # Calculate bin volume if it has not been calculated yet
        if (r_ind, t_ind, p_ind) not in bin_volumes:
            bin_volumes[(r_ind, t_ind, p_ind)] = get_sect_vol(
                    (rvals[r_ind], rvals[r_ind+1]),
                    (tvals[t_ind], tvals[t_ind+1]),
                    (pvals[p_ind], pvals[p_ind+1]))

        # Update bin weight
        orig_weight = 0
        if (r_ind, t_ind, p_ind) in bin_weights:
            orig_weight = bin_weights[(r_ind, t_ind, p_ind)]
        new_pt_weight = 1 / (pt_density * bin_volumes[(r_ind, t_ind, p_ind)]**(1/3.0))
        bin_weights[(r_ind, t_ind, p_ind)] = orig_weight + new_pt_weight

    sc_features = np.zeros(R_BINS * T_BINS * P_BINS)
    r_ind = 0 if R_BINS == 1 else T_BINS*P_BINS
    p_ind = 0 if P_BINS == 1 else T_BINS

    for (r, t, p) in bin_weights.iterkeys():
        sc_features[r*r_ind + p*p_ind + t] = bin_weights[(r, t, p)]
    return sc_features

def extract_point(hmat):
    return hmat[:3, 3]

def hmat_cost(hmat1, hmat2):
    pt1, pt2 = extract_point(hmat1), extract_point(hmat2)
    return np.linalg.norm(pt1 - pt2)

def compare_hmats(traj1, traj2):
    """
    Uses Dynamic Time Warping to compare sequences of gripper poses 

     -- currently ignores the poses, just looks at positions along trajcetory
    """
    n = len(traj1)
    m = len(traj2)
    DTW = np.zeros((n+1, m+1))
    DTW[:,0] = np.inf
    DTW[0,:] = np.inf
    DTW[0, 0] = 0
    for i, hmat1 in enumerate(traj1):
        i = i+1 # increase by one because we need to 1 index DTW
        for j, hmat2 in enumerate(traj2):
            j = j+1 
            best_next = min(DTW[i-1, j], DTW[i, j-1], DTW[i-1, j-1])
            DTW[i, j] = hmat_cost(hmat1, hmat2) + best_next
    return DTW[n, m]/float(max(n, m))

def warp_hmats(xyz_src, xyz_targ, hmat_list, src_interest_pts = None):
    f, src_params, g, targ_params, cost = registration_cost(xyz_src, xyz_targ, src_interest_pts)
    f = registration.unscale_tps(f, src_params, targ_params)
    trajs = {}
    for k, hmats in hmat_list:
        trajs[k] = f.transform_hmats(hmats)
    xyz_src_warped = f.transform_points(xyz_src)
    return [trajs, cost, xyz_src_warped]

def compute_weights(xyz, interest_pts):
    radius = np.max(ssd.cdist(xyz, xyz, 'euclidean'))/10.0
    distances = np.exp(-np.min(ssd.cdist(xyz, interest_pts, 'euclidean'), axis=1)/radius)
    return 1+distances

def registration_cost(xyz_src, xyz_targ, src_interest_pts=None):
    if src_interest_pts is not None:
        weights = compute_weights(xyz_src, src_interest_pts)
    else:
        weights = None
    scaled_xyz_src, src_params = registration.unit_boxify(xyz_src)
    scaled_xyz_targ, targ_params = registration.unit_boxify(xyz_targ)
    f,g = registration.tps_rpm_bij(scaled_xyz_src, scaled_xyz_targ, plot_cb=None,
                                   plotting=0, rot_reg=np.r_[1e-4, 1e-4, 1e-1], 
                                   n_iter=50, reg_init=10, reg_final=.1, outlierfrac=1e-2,
                                   x_weights=weights)
    cost = registration.tps_reg_cost(f) + registration.tps_reg_cost(g)
    return f, src_params, g, targ_params, cost

def registration_cost_cheap(xyz0, xyz1):
    scaled_xyz0, _ = registration.unit_boxify(xyz0)
    scaled_xyz1, _ = registration.unit_boxify(xyz1)
    f,g = registration.tps_rpm_bij(scaled_xyz0, scaled_xyz1, rot_reg=1e-3, n_iter=10)
    cost = registration.tps_reg_cost(f) + registration.tps_reg_cost(g)
    return cost

def warp_hmats_parallel_precompute(act_set, states, actions, ppservers=()):
    scaled_xyz_srcs = []
    srcs_params = []
    for action in actions:
        scaled_xyz_src, src_params = registration.unit_boxify(act_set.actions_ds_clouds[action])
        scaled_xyz_srcs.append(scaled_xyz_src)
        srcs_params.append(src_params)
    scaled_xyz_srcs = np.array(scaled_xyz_srcs)
    scaled_xyz_targs = []
    targs_params = []
    for state in states:
        scaled_xyz_targ, targ_params = registration.unit_boxify(state[1])
        scaled_xyz_targs.append(scaled_xyz_targ)
        targs_params.append(targ_params)
    scaled_xyz_targs = np.array(scaled_xyz_targs)
    
    tps_tups = tps_registration_parallel.tps_rpm_bij_grid(scaled_xyz_srcs, scaled_xyz_targs, plot_cb=None,
                                                        plotting=0, rot_reg=np.r_[1e-4, 1e-4, 1e-1], 
                                                        n_iter=50, reg_init=10, reg_final=.1, outlierfrac=1e-2,
                                                        parallel=True, ppservers=ppservers, partition_step=10)
    for (i,action) in enumerate(actions):
        hmat_list = [(lr, act_set.actionfile[action][ln]['hmat']) for lr, ln in zip('lr', act_set.link_names)]
        for (j,state) in enumerate(states):
            key = (state[0], action)
            f,g = tps_tups[i,j]
            cost = registration.tps_reg_cost(f) + registration.tps_reg_cost(g)
            f = registration.unscale_tps(f, srcs_params[i], targs_params[j])
            trajs = {}
            for k, hmats in hmat_list:
                trajs[k] = f.transform_hmats(hmats)
            xyz_src_warped = f.transform_points(act_set.actions_ds_clouds[action])
            if key in act_set.caches['warp_hmats']:
                print "Warning: warp_hmats_parallel_precompute is overriding values"
            act_set.caches['warp_hmats'][key] = [trajs, cost, xyz_src_warped]

def registration_cost_cheap_parallel_precompute(act_set, states, actions, ppservers=()):
    scaled_xyz_srcs = []
    for action in actions:
        scaled_xyz_src, src_params = registration.unit_boxify(act_set.actions_ds_clouds[action])
        scaled_xyz_srcs.append(scaled_xyz_src)
    scaled_xyz_srcs = np.array(scaled_xyz_srcs)
    scaled_xyz_targs = []
    for state in states:
        scaled_xyz_targ, targ_params = registration.unit_boxify(state[1])
        scaled_xyz_targs.append(scaled_xyz_targ)
    scaled_xyz_targs = np.array(scaled_xyz_targs)

    tps_tups = tps_registration_parallel.tps_rpm_bij_grid(scaled_xyz_srcs, scaled_xyz_targs, rot_reg=1e-3, n_iter=10, parallel=True, ppservers=ppservers, partition_step=10)
    for (i,action) in enumerate(actions):
        for (j,state) in enumerate(states):
            key = (state[0], action)
            if key in act_set.caches['cheap_reg_costs']:
                print "Warning: registration_cost_cheap_parallel_precompute is overriding values"
            act_set.caches['cheap_reg_costs'][key] = registration.tps_reg_cost(tps_tups[i,j][0]) + registration.tps_reg_cost(tps_tups[i,j][1])

def landmark_features_parallel_precompute(act_set, states, ppservers=()):
    scaled_xyz_srcs = []
    for i in range(len(act_set.landmarks)):
        landmark = act_set.landmarks[str(i)]
        scaled_xyz_src, src_params = registration.unit_boxify(landmark['cloud_xyz'][()])
        scaled_xyz_srcs.append(scaled_xyz_src)
    scaled_xyz_srcs = np.array(scaled_xyz_srcs)
    scaled_xyz_targs = []
    for state in states:
        scaled_xyz_targ, targ_params = registration.unit_boxify(state[1])
        scaled_xyz_targs.append(scaled_xyz_targ)
    scaled_xyz_targs = np.array(scaled_xyz_targs)

    tps_tups = tps_registration_parallel.tps_rpm_bij_grid(scaled_xyz_srcs, scaled_xyz_targs, rot_reg=1e-3, n_iter=10, parallel=True, ppservers=ppservers, partition_step=10)
    for (j,state) in enumerate(states):
        feat = np.empty(len(act_set.landmarks))
        for i in range(len(act_set.landmarks)):
            feat[i] = registration.tps_reg_cost(tps_tups[i,j][0]) + registration.tps_reg_cost(tps_tups[i,j][1])
        key = state[0]
        if key in act_set.caches['landmarks']:
            print "Warning: landmark_features_parallel_precompute is overriding values"
        act_set.caches['landmarks'][key] = feat

def combine_expert_demo_files(infile1, infile2, outfile):
    """
    takes the data from the two specified files and returns
    a new file that contains the examples from both files
    """
    try:
        if1 = h5py.File(infile1, 'r')
        if2 = h5py.File(infile2, 'r')
        of = h5py.File(outfile, 'w')
        values = if1.values() + if2.values()
        for (i, in_g) in enumerate(values):
            if 'action' not in in_g or 'cloud_xyz' not in in_g:
                print "file format incorrect, skipping entry"
                continue
            out_g = of.create_group(str(i))
            out_g['action'] = in_g['action'][()]
            out_g['cloud_xyz'] = in_g['cloud_xyz'][:]
    finally:
        if1.close()
        if2.close()
        of.close()

def get_traj_pts(seg_info):
    """
    Returns locations of the gripper at all T time steps of the trajectory.
    Returns 2*T x 3 Numpy array, with first T rows corresponding to left
    gripper's positions and the next T corresponding to the right gripper's.
    """
    traj_pts = []
    for lr in "lr":
        hmats = seg_info["%s_gripper_tool_frame"%lr]['hmat']
        for hmat in hmats:
            traj_pts.append(extract_point(hmat))
    return np.array(traj_pts)    

def get_closing_pts(seg_info, as_dict = False):
    closing_inds = get_closing_inds(seg_info)
    closing_pts = []
    if as_dict: closing_pts = {}
    for lr in closing_inds:
        if closing_inds[lr] != -1:
            hmat = seg_info["%s_gripper_tool_frame"%lr]['hmat'][closing_inds[lr]]
            if not as_dict:
                closing_pts.append(extract_point(hmat))
            else:
                closing_pts[lr] = extract_point(hmat)
    return closing_pts

def get_closing_inds(seg_info):
    """
    returns a dictionary mapping 'l', 'r' to the index in the corresponding trajectory
    where the gripper first closes
    """
    result = {}
    for lr in 'lr':
        grip = np.asarray(seg_info[lr + '_gripper_joint'])
        closings = np.flatnonzero((grip[1:] < GRIPPER_OPEN_CLOSE_THRESH) \
                                      & (grip[:-1] >= GRIPPER_OPEN_CLOSE_THRESH))
        if closings:
            result[lr] = closings[0]
        else:
            result[lr] = -1
    return result

def compute_action_margin(model, a1, a2):
    print 'done'
    return model.margin(None, a1, a2)

def bellman_test_features(args):
    if args.model != 'bellman':
        raise Exception, 'wrong model for this'
    act_set = ActionSet(args.actionfile, landmarks=args.landmark_features, gripper_weighting=args.gripper_weighting)
    feature_fn, margin_fn, num_features = select_feature_fn(args, act_set)
    demofile = h5py.File(args.demofile, 'r')
    # get a random set of trajectories
    trajectories = []
    traj = []
    for uid in range(len(demofile)):
        key = str(uid)
        group = demofile[key]
        state = [key,group['cloud_xyz'][:]] # these are already downsampled
        action = group['action'][()] if not group['knot'][()] else 'done'
        if group['pred'][()] == key:
            if traj:
                trajectories.append(traj)
                traj = []
        traj.append([state, action])
    if traj:
        trajectories.append(traj)
    random.shuffle(trajectories)
    constraint_trajs = trajectories[:args.num_constraints]
    # put them into a Bellman model
    mm_model = BellmanMaxMarginModel(action, 500, 1000, 10, .9, num_features, feature_fn, margin_fn)
    for i, t in enumerate(constraint_trajs):
        mm_model.add_trajectory(t, str(i), True)
    weights, w0 = mm_model.optimize_model()
    # evaluate value fn performance
    num_evals = len(trajectories) if args.num_evals < 0 else args.num_constraints + args.num_evals
    trajectories = trajectories[args.num_constraints:num_evals]
    values = np.zeros((len(trajectories), 6))
    num_decreases = 0
    for (i, t) in enumerate(trajectories):
        for j, (s, a) in enumerate(t):
            values[i, j] = np.dot(mm_model.weights, mm_model.feature_fn(s, a))
        for k in range(j):
            if values[i, k] > values[i, k+1]: 
                num_decreases += 1
                print "DECREASE", values[i, k], values[i, k+1]
        sys.stdout.write('num decreases:\t{} computed values for trajectory {}\r'.format(num_decreases, i))
        sys.stdout.flush()
    sys.stdout.write('\n')
    print "num decreases:\t", num_decreases
    ipy.embed()    

def test_saving_model(mm_model):
    # Use Gurobi to save the model in MPS format
    weights, w0 = mm_model.optimize_model()
    mm_model.save_model('data/rope_model_saved_test.mps')
    mm_model_saved = grb.read('data/rope_model_saved_test.mps')
    mm_model_saved.optimize()
    weights_from_saved = [x.X for x in mm_model.w]

    saved_correctly = True
    for i in range(len(weights)):
        if weights[i] != weights_from_saved[i]:
            print "ERROR in saving model: Orig weight: ", weights[i], \
                  ", Weight from optimizing saved model: ", weights_from_saved[i]
            saved_correctly = False
    if saved_correctly:
        print "PASSED: Model saved and reloaded correctly"

def select_feature_fn(args, act_set):
    if args.ensemble:
        print 'Using bias, quad, sc, ropedist, landmark ({}), done, is_knot features, traj_diff.'.format(args.landmark_features)
        curried_done_fn = lambda act_set: get_done_feature_fn(act_set)
        curried_landmark_fn = lambda act_set: get_landmark_feature_fn(act_set, rbf=args.rbf)
        fns = [get_quad_feature_fn, get_sc_feature_fn, get_rope_dist_feat_fn,
               curried_landmark_fn]
        if args.traj_features:
            fns.append(get_traj_diff_feature_fn)
        feature_fn, num_features = concatenate_fns(fns, act_set)
    elif args.quad_landmark_features and args.landmark_features:
        print 'Using bias, quad, landmark ({}) features.'.format(args.landmark_features)
        curried_landmark_fn = lambda act_set: get_landmark_feature_fn(act_set, rbf=args.rbf)
        fns = [get_quad_feature_noregcostsq_fn, curried_landmark_fn]
        feature_fn, num_features = concatenate_fns(fns, act_set)
    elif args.landmark_features and not args.only_landmark:
        print 'Using bias, quad, sc, ropedist, landmark ({}) features.'.format(args.landmark_features)
        curried_landmark_fn = lambda act_set: get_landmark_feature_fn(act_set, rbf=args.rbf)
        fns = [get_quad_feature_fn, get_sc_feature_fn, get_rope_dist_feat_fn,
               curried_landmark_fn]
        feature_fn, num_features = concatenate_fns(fns, act_set)
    elif args.landmark_features:
        print 'Using landmark {} features'.format(args.landmark_features)
        feature_fn, num_features = get_landmark_feature_fn(act_set, rbf=args.rbf)
    elif args.quad_features:
        print 'Using quadratic features.'
        feature_fn, num_features = get_quad_feature_fn(act_set)
    elif args.rope_dist_features:
        print 'Using sc, bias, and rope dist features.'
        fns = [get_bias_feature_fn, get_sc_feature_fn, get_rope_dist_feat_fn]
        feature_fn, num_features = concatenate_fns(fns, act_set)
    elif args.sc_features:
        print 'Using sc and bias features.'
        fns = [get_bias_feature_fn, get_sc_feature_fn]
        feature_fn, num_features = concatenate_fns(fns, act_set)
    else:
        print 'Using bias features.'
        feature_fn, num_features = get_bias_feature_fn(act_set)
    margin_fn = get_action_state_margin_fn(act_set)
    return feature_fn, margin_fn, num_features

def build_constraints_no_model(args):
    act_set = ActionSet(args.actionfile, landmarks=args.landmark_features, gripper_weighting=args.gripper_weighting)
    feature_fn, margin_fn, num_features = select_feature_fn(args, act_set)
    print 'Building constraints using no model into {}.'.format(args.constraintfile)
    if args.model == 'bellman':
        compute_bellman_constraints_no_model(feature_fn,
                                             margin_fn,
                                             act_set,
                                             args.demofile,
                                             outfile=args.constraintfile,
                                             start=args.start,
                                             end=args.end,
                                             verbose=True,
                                             parallel=args.parallel,
                                             ppservers=tuple(args.ppservers),
                                             ignore_bellman_constraints=args.ignore_bellman_constraints)
    else:
        if args.ignore_bellman_constraints:
            raise RuntimeError('Option ignore_bellman_constraints is incompatible with non-bellan model')
        compute_constraints_no_model(feature_fn,
                                     margin_fn,
                                     act_set,
                                     args.demofile,
                                     outfile=args.constraintfile,
                                     start=args.start,
                                     end=args.end,
                                     verbose=True)

def build_constraints(args):
    act_set = ActionSet(args.actionfile, landmarks=args.landmark_features, gripper_weighting=args.gripper_weighting)
    feature_fn, margin_fn, num_features = select_feature_fn(args, act_set)
    print 'Building constraints into {}.'.format(args.constraintfile)
    if args.model == 'multi':
        mm_model = MultiSlackMaxMarginModel(act_set.actions, num_features, feature_fn, margin_fn)
    elif args.model == 'bellman':
        mm_model = BellmanMaxMarginModel(act_set.actions, .9, num_features, feature_fn, margin_fn)
    else:
        mm_model = MaxMarginModel(act_set.actions, num_features, feature_fn, margin_fn)
    if args.model == 'bellman':
        add_bellman_constraints_from_demo(mm_model,
                                          args.demofile,
                                          args.start, args.end,
                                          outfile=args.constraintfile,
                                          verbose=True)
    else:
        add_constraints_from_demo(mm_model,
                                  args.demofile,
                                  args.start, args.end,
                                  outfile=args.constraintfile,
                                  verbose=True)

def build_model(args):
    act_set = ActionSet(args.actionfile, landmarks=args.landmark_features, gripper_weighting=args.gripper_weighting)
    feature_fn, margin_fn, num_features = select_feature_fn(args, act_set)
    print 'Building model into {}.'.format(args.modelfile)
    if args.model == 'multi':
        mm_model = MultiSlackMaxMarginModel(act_set.actions, num_features, feature_fn, margin_fn)
    elif args.model == 'bellman':
        mm_model = BellmanMaxMarginModel(act_set.actions, 1, num_features, feature_fn, margin_fn) # changed
    else:
        mm_model = MaxMarginModel(act_set.actions, num_features, feature_fn, margin_fn)
    mm_model.load_constraints_from_file(args.constraintfile)
    mm_model.save_model(args.modelfile)

def build_model_and_merge(args):
    act_set = ActionSet(args.actionfile, landmarks=args.landmark_features, gripper_weighting=args.gripper_weighting)
    feature_fn, margin_fn, num_features = select_feature_fn(args, act_set)
    print 'Found unmerged model: {}'.format(args.unmerged_modelfile)
    print 'Building merged model into {}.'.format(args.modelfile)
    if args.model == 'multi':
        mm_model = MultiSlackMaxMarginModel.read(args.unmerged_modelfile, act_set.actions, num_features, feature_fn, margin_fn)
    elif args.model == 'bellman':
        mm_model = BellmanMaxMarginModel.read(args.unmerged_modelfile, act_set.actions, num_features, feature_fn, margin_fn)
    else:
        mm_model = MaxMarginModel.read(args.unmerged_modelfile, act_set.actions, num_features, feature_fn, margin_fn)
    constraintfile_base_noext = os.path.splitext(os.path.split(args.constraintfile)[-1])[0]
    mm_model.load_constraints_from_file(args.constraintfile, slack_name_postfix="_"+constraintfile_base_noext)
    mm_model.save_model(args.modelfile)

def optimize_model(args):
    act_set = ActionSet(args.actionfile, landmarks=args.landmark_features, gripper_weighting=args.gripper_weighting)
    feature_fn, margin_fn, num_features = select_feature_fn(args, act_set)
    print 'Found model: {}'.format(args.modelfile)
    if args.model == 'multi':
        mm_model = MultiSlackMaxMarginModel.read(args.modelfile, act_set.actions, num_features, feature_fn, margin_fn)
        mm_model.scale_objective(args.C)
    elif args.model == 'bellman':
        mm_model = BellmanMaxMarginModel.read(args.modelfile, act_set.actions, num_features, feature_fn, margin_fn)
        mm_model.scale_objective(args.C, args.D, args.F)
    else:
        mm_model = MaxMarginModel.read(args.modelfile, act_set.actions, num_features, feature_fn, margin_fn)
        mm_model.scale_objective(args.C)
    if args.save_memory:
        mm_model.model.setParam('threads', 1)  # Use single thread instead of maximum
        # barrier method (#2) is default for QP, but uses more memory and could lead to error
        #mm_model.model.setParam('method', 1)  # Use dual simplex method to solve model
        mm_model.model.setParam('method', 0)  # Use primal simplex method to solve model
    mm_model.optimize_model()
    mm_model.save_weights_to_file(args.weightfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    parser.add_argument('model', choices=['single', 'multi', 'bellman'])
    parser.add_argument('--ensemble', action='store_true')
    parser.add_argument('--landmark_features')
    parser.add_argument('--quad_landmark_features', action='store_true')
    parser.add_argument('--only_landmark', action="store_true")
    parser.add_argument('--rbf', action='store_true')
    parser.add_argument("--quad_features", action="store_true")
    parser.add_argument("--sc_features", action="store_true")
    parser.add_argument("--rope_dist_features", action="store_true")
    parser.add_argument("--traj_features", action="store_true")
    parser.add_argument("--save_memory", action="store_true")
    parser.add_argument("--gripper_weighting", action="store_true")
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--ppservers', type=str, nargs='*', default=[])

    # bellman test subparser
    parser_test_bellman = subparsers.add_parser('test-bellman')
    parser_test_bellman.add_argument('actionfile', nargs='?', default='data/misc/actions.h5')
    parser_test_bellman.add_argument('demofile')
    parser_test_bellman.add_argument("--num_constraints", type=int, default = 20)
    parser_test_bellman.add_argument("--num_evals", type=int, default = -1) # defaults to all
    parser_test_bellman.set_defaults(func=bellman_test_features)

    # build-constraints-no-model subparser
    parser_build_constraints = subparsers.add_parser('build-constraints-no-model')
    parser_build_constraints.add_argument('demofile')
    parser_build_constraints.add_argument('constraintfile')
    parser_build_constraints.add_argument('--start', type=int, default=0)
    parser_build_constraints.add_argument('--end', type=int, default=-1)
    parser_build_constraints.add_argument('--ignore_bellman_constraints', action='store_true')
    parser_build_constraints.add_argument('actionfile', nargs='?', default='data/misc/actions.h5')
    parser_build_constraints.set_defaults(func=build_constraints_no_model)
    
    # build-constraints subparser
    parser_build_constraints = subparsers.add_parser('build-constraints')
    parser_build_constraints.add_argument('demofile')
    parser_build_constraints.add_argument('constraintfile')
    parser_build_constraints.add_argument('--start', type=int, default=0)
    parser_build_constraints.add_argument('--end', type=int, default=-1)
    parser_build_constraints.add_argument('actionfile', nargs='?', default='data/misc/actions.h5')
    parser_build_constraints.set_defaults(func=build_constraints)

    # build-model subparser
    parser_build_model = subparsers.add_parser('build-model')
    parser_build_model.add_argument('constraintfile')
    parser_build_model.add_argument('modelfile')
    parser_build_model.add_argument('actionfile', nargs='?', default='data/misc/actions.h5')
    parser_build_model.set_defaults(func=build_model)

    # build-model-merge subparser
    parser_build_model = subparsers.add_parser('build-model-merge')
    parser_build_model.add_argument('constraintfile')
    parser_build_model.add_argument('unmerged_modelfile')
    parser_build_model.add_argument('modelfile')
    parser_build_model.add_argument('actionfile', nargs='?', default='data/misc/actions.h5')
    parser_build_model.set_defaults(func=build_model_and_merge)

    # optimize-model subparser
    parser_optimize = subparsers.add_parser('optimize-model')
    parser_optimize.add_argument('--C', '-c', type=float, default=1)
    parser_optimize.add_argument('--D', '-d', type=float, default=1)
    parser_optimize.add_argument('--F', '-f', type=float, default=1)
    parser_optimize.add_argument('modelfile')
    parser_optimize.add_argument('weightfile')
    parser_optimize.add_argument('actionfile', nargs='?', default='data/misc/actions.h5')
    parser_optimize.set_defaults(func=optimize_model)

    # parse args and call appropriate function
    args = parser.parse_args()
    args.func(args)
