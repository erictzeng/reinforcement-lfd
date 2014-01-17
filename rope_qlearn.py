#!/usr/bin/env python

"""
code for doing max margin Qlearning for knot-tying
includes helper functions for manipulating demonstration
and action data
"""

import argparse

import h5py, math
import gurobipy as grb
import IPython as ipy
from max_margin import MaxMarginModel, MultiSlackMaxMarginModel
from pdb import pm
import numpy as np
from joblib import Parallel, delayed
import scipy.spatial as sp_spat
import os.path
import cProfile
try:
    from rapprentice import registration, clouds
    use_rapprentice = True
except:
    print "Couldn't import from rapprentice"
    # set a flag so we can test some stuff without that functionality
    use_rapprentice = False

DS_SIZE = .025
GRIPPER_OPEN_CLOSE_THRESH = 0.04

# constants for shape context
R_MIN, P_MIN, T_MIN = 0, 0, np.pi/2.0
R_MAX, P_MAX, T_MAX = 0.15, np.pi, np.pi
R_BINS, P_BINS, T_BINS = 2, 4, 4
DENSITY_RADIUS = .2


# rip rope_max_margin_model, you will not be missed --eric 1/14/2014

def add_constraints_from_demo(mm_model, expert_demofile, outfile=None, verbose=False):
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
    max = float('inf')
    c = 0
    for key, group in expert_demofile.iteritems():
        if c > max: break
        state = [key,group['cloud_xyz'][:]] # these are already downsampled
        action = group['action'][()]
        if action.startswith('endstate'): # this is a knot
            continue
        if verbose:
            print 'adding constraints for:\t', action
        c += 1
        mm_model.add_example(state, action, verbose)
        #mm_model.clear_asm_cache()
        if outfile:
            mm_model.save_constraints_to_file(outfile)

def concatenate_fns(fns, actionfile):
    if type(actionfile) is str:
        actionfile = h5py.File(actionfile, 'r')
    
    fn_params = [f(actionfile) for f in fns]
    def result_fn(state, action):
        results = np.array([])
        for (f, _, _) in fn_params:
            results = np.r_[results, f(state, action)]
        return results

    N = sum(v[1] for v in fn_params)
    return (result_fn, N, actionfile)

def get_sc_feature_fn(actionfile):
    if type(actionfile) is str:
        actionfile = h5py.File(actionfile, 'r')
    act_set = ActionSet(actionfile)
    return (act_set.sc_features, act_set.num_sc_features, actionfile)    

def get_bias_feature_fn(actionfile, old=False):
    if type(actionfile) is str:
        actionfile = h5py.File(actionfile, 'r')
    act_set = ActionSet(actionfile)
    def feature_fn(state, action):
        return act_set.bias_features(state, action, old)
    return (feature_fn, act_set.num_actions + 1, actionfile)

def get_quad_feature_fn(actionfile, old=False):
    if type(actionfile) is str:
        actionfile = h5py.File(actionfile, 'r')
    act_set = ActionSet(actionfile)
    def feature_fn(state, action):
        return act_set.quad_features(state, action, old)
    return (feature_fn, 2 + 2*act_set.num_actions, actionfile)

def get_action_only_margin_fn(actionfile):
    if type(actionfile) is str:
        actionfile = h5py.File(actionfile, 'r')
    act_set = ActionSet(actionfile)
    return (act_set.action_only_margin, actionfile)

def get_action_state_margin_fn(actionfile):
    if type(actionfile) is str:
        actionfile = h5py.File(actionfile, 'r')
    act_set = ActionSet(actionfile)
    return (act_set.action_state_margin, actionfile)

class ActionSet(object):
    """
    class to handle computing features and margins for state/action pairs


    state is assumed to be a list [<state_id>, <point_cloud>]
    """
    def __init__(self, actionfile, use_cache = True):
        self.actionfile = actionfile
        self.actions = sorted(actionfile.keys())
        self.action_to_ind = dict((v, i) for i, v in enumerate(self.actions))
        self.num_actions = len(self.actions)
        self.num_sc_features = R_BINS*T_BINS*P_BINS*2
        self.cache = {}
        self.use_cache = use_cache
        self.link_names = ["%s_gripper_tool_frame"%lr for lr in ('lr')]

    def _warp_hmats(self, state, action):
        hit, value = self.check_cache(state, action)
        if hit:
            return value
        else:
            [warped_trajs, rc] = warp_hmats(self.get_ds_cloud(action),
                                      state[1],
                                      [(lr, self.actionfile[action][ln]['hmat']) for lr, ln in zip('lr', self.link_names)])
            self.store_cache(state, action, [warped_trajs, rc])
            return [warped_trajs, rc]

    def get_ds_cloud(self, action):
        return clouds.downsample(self.actionfile[action]['cloud_xyz'], DS_SIZE)

    def sc_features(self, state, action):
        seg_info = self.actionfile[action]

        warped_trajs, _ = self._warp_hmats(state, action)
        feat_val = dict((lr, np.zeros(self.num_sc_features/2.0)) for lr in 'lr')
        for lr in 'lr':
            grip = np.asarray(seg_info[lr + '_gripper_joint'])
            closings = np.flatnonzero((grip[1:] < GRIPPER_OPEN_CLOSE_THRESH) & (grip[:-1] >= GRIPPER_OPEN_CLOSE_THRESH))
            if closings:
                first_close = closings[0]
                close_hmat = warped_trajs[lr][first_close]
                feat_val[lr] = gripper_frame_shape_context(state[1], close_hmat)
        return np.r_[feat_val['l'], feat_val['r']]            
    
    def bias_features(self, state, action, old = False):
        feat = np.zeros(self.num_actions + 1)
        if old:
            feat[0] = registration_cost(state[1], self.get_ds_cloud(action))
        else:
            (_, feat[0]) = self._warp_hmats(state, action)
        feat[self.action_to_ind[action]+1] = 1
        return feat
    
    def quad_features(self, state, action, old = False):
        feat = np.zeros(2 + 2*self.num_actions)
        if old:
            s = registration_cost(state[1], self.get_ds_cloud(action))
        else:
            (_, s) = self._warp_hmats(state, action)
        feat[0] = s**2
        feat[1] = s
        feat[2+self.action_to_ind[action]] = s
        feat[2+self.num_actions+self.action_to_ind[action]] = 1
        return feat

    def action_only_margin(self, s, a1, a2):
        """
        warp both actions, compare the resulting trajectories:
        ex. warp a1 -> a2; use compare_hmats(warp(a1.traj), a2.traj)
        """
        warped_a1_trajs = self._warp_hmats((a2, self.get_ds_cloud(a2)), a1)
        warped_a1_trajs = [warped_a1_trajs[lr] for lr in 'lr']
        a1_trajs = [self.actionfile[a1][ln]['hmat'][:] for ln in self.link_names]
        warped_a2_trajs = self._warp_hmats((a1, self.get_ds_cloud(a1)), a2)
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
        warped_a1_trajs, _ = self._warp_hmats(state, a1)
        warped_a2_trajs, _ = self._warp_hmats(state, a2)
        return sum(compare_hmats(warped_a1_trajs[lr], warped_a2_trajs[lr]) for lr in 'lr')

    def combined_margin(self, state, a1, a2):
        return (self.action_only_margin(a1, a2) +
                self.action_state_margin(a1, a2, state))

#http://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
def cart2spherical(xyz):
    xyz = np.asarray(xyz)
    ptsnew =  np.zeros(xyz.shape)
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,0] = np.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,1] = np.arctan2(xyz[:,1], xyz[:,0])
    ptsnew[ptsnew[:,1] < 0] += 2*np.pi
    ptsnew[:,2] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    return ptsnew

def get_sect_vol((r1, r2), (t1, t2), (p1, p2)):
    return (r2**3 - r1**3)/3 * (t2 - t1) * (-math.cos(p2) + math.cos(p1))

def gripper_frame_shape_context(xyz, hmat):
    h_inv = np.linalg.inv(hmat)
    xyz1 = np.ones((len(xyz),4),'float')  #homogeneous coord
    xyz1[:,:3] = xyz
    xyz2 = [np.dot(h_inv, pt)[:3] for pt in xyz1] #bestpractices
    xyz3 = cart2spherical(xyz2)
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
    for (r, p, t) in bin_weights.iterkeys():
        sc_features[r*T_BINS*P_BINS + p*T_BINS + t] = bin_weights[(r, p, t)]
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
    return DTW[n, m]

def warp_hmats(xyz_src, xyz_targ, hmat_list):
    if not use_rapprentice:
        return hmat_list
    scaled_xyz_src, src_params = registration.unit_boxify(xyz_src)
    scaled_xyz_targ, targ_params = registration.unit_boxify(xyz_targ)        
    f,g = registration.tps_rpm_bij(scaled_xyz_src, scaled_xyz_targ, plot_cb = None,
                                   plotting=0,rot_reg=np.r_[1e-4,1e-4,1e-1], 
                                   n_iter=50, reg_init=10, reg_final=.1)
    cost = registration.tps_reg_cost(f) + registration.tps_reg_cost(g)
    f = registration.unscale_tps(f, src_params, targ_params)
    trajs = {}
    for k, hmats in hmat_list:
        trajs[k] = f.transform_hmats(hmats)
    return [trajs, cost]

def get_downsampled_clouds(demofile):
    if not use_rapprentice:
        return get_clouds(demofile)
    return [clouds.downsample(seg["cloud_xyz"], DS_SIZE) for seg in demofile.values()]

def get_clouds(demofile):
    return [seg["cloud_xyz"] for seg in demofile.values()]

def registration_cost(xyz0, xyz1):
    if not use_rapprentice:
        return 1
    scaled_xyz0, _ = registration.unit_boxify(xyz0)
    scaled_xyz1, _ = registration.unit_boxify(xyz1)
    f,g = registration.tps_rpm_bij(scaled_xyz0, scaled_xyz1, rot_reg=1e-3, n_iter=10)
    cost = registration.tps_reg_cost(f) + registration.tps_reg_cost(g)
    return cost

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

def compute_action_margin(model, a1, a2):
    print 'done'
    return model.margin(None, a1, a2)

def test_saving_model(mm_model):
    # Use Gurobi to save the model in MPS format
    weights = mm_model.optimize_model()
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

def select_feature_fn(args):
    if args.quad_features:
        print 'Using quadratic features.'
        feature_fn, num_features, act_file = get_quad_feature_fn(args.actionfile)
    elif args.sc_features:
        print 'Using sc and bias features.'
        fns = [get_bias_feature_fn, get_sc_feature_fn]
        feature_fn, num_features, act_file = concatenate_fns(fns, args.actionfile)
    else:
        print 'Using bias features.'
        feature_fn, num_features, act_file = get_bias_feature_fn(args.actionfile)
    margin_fn, act_file = get_action_state_margin_fn(act_file)
    actions = act_file.keys()
    return feature_fn, margin_fn, num_features, actions

def test_sc_features(args):
    feature_fn, num_features, act_file = get_sc_feature_fn(args.actionfile)
    for name, seg_info in act_file.iteritems():
        print feature_fn([name, clouds.downsample(seg_info['cloud_xyz'], DS_SIZE)], name)

def build_constraints(args):
    feature_fn, margin_fn, num_features, actions = select_feature_fn(args)
    print 'Building constraints into {}.'.format(args.constraintfile)
    if args.multi_slack:
        mm_model = MultiSlackMaxMarginModel(actions, args.C, num_features, feature_fn, margin_fn)
    else:
        mm_model = MaxMarginModel(actions, args.C, num_features, feature_fn, margin_fn)
    add_constraints_from_demo(mm_model,
                              args.demofile,
                              outfile=args.constraintfile,
                              verbose=True)

def build_model(args):
    feature_fn, margin_fn, num_features, actions = select_feature_fn(args)
    print 'Building model into {}.'.format(args.modelfile)
    if args.multi_slack:
        mm_model = MultiSlackMaxMarginModel(actions, args.C, num_features, feature_fn, margin_fn)
    else:
        mm_model = MaxMarginModel(actions, args.C, num_features, feature_fn, margin_fn)
    mm_model.load_constraints_from_file(args.constraintfile)
    mm_model.save_model(args.modelfile)

def optimize_model(args):
    feature_fn, margin_fn, num_features, actions = select_feature_fn(args)
    print 'Found model: {}'.format(args.modelfile)
    if args.multi_slack:
        mm_model = MultiSlackMaxMarginModel.read(args.modelfile, actions, feature_fn, margin_fn)
    else:
        mm_model = MaxMarginModel.read(args.modelfile, actions, feature_fn, margin_fn)
    mm_model.C = args.C
    mm_model.optimize_model()
    mm_model.save_weights_to_file(args.weightfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    parser.add_argument("--quad_features", action="store_true")
    parser.add_argument("--sc_features", action="store_true")
    parser.add_argument('--C', '-c', type=float, default=1)
    parser.add_argument("--multi_slack", action="store_true")

    # build-constraints subparser
    parser_build_constraints = subparsers.add_parser('build-constraints')
    parser_build_constraints.add_argument('demofile')
    parser_build_constraints.add_argument('constraintfile')
    parser_build_constraints.add_argument('actionfile', nargs='?', default='data/all.h5')
    parser_build_constraints.set_defaults(func=build_constraints)

    # build-model subparser
    parser_build_model = subparsers.add_parser('build-model')
    parser_build_model.add_argument('constraintfile')
    parser_build_model.add_argument('modelfile')
    parser_build_model.add_argument('actionfile', nargs='?', default='data/all.h5')
    parser_build_model.set_defaults(func=build_model)

    # optimize-model subparser
    parser_optimize = subparsers.add_parser('optimize-model')
    parser_optimize.add_argument('modelfile')
    parser_optimize.add_argument('weightfile')
    parser_optimize.add_argument('actionfile', nargs='?', default='data/all.h5')
    parser_optimize.set_defaults(func=optimize_model)

    # parse args and call appropriate function
    args = parser.parse_args()
    args.func(args)
