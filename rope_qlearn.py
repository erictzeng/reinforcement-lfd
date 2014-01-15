#!/usr/bin/env python

"""
code for doing max margin Qlearning for knot-tying
includes helper functions for manipulating demonstration
and action data
"""

import argparse

import h5py
import gurobipy as grb
import IPython as ipy
from max_margin import MaxMarginModel
from pdb import pm
import numpy as np
from joblib import Parallel, delayed
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

def rope_max_margin_model(actionfile, C, N, feature_fn, margin_fn, constraints = None):    
    """
    returns a handle to an optimization model that comptues uses a max margin
    approach to learn a q function
    can include use precomputed constraints if specified
    """
    actions = actionfile.keys() # feature_fn takes a seg name and 
    # computes appropriately (so we don't spend a bunch of time copying over
    # clouds and hmat lists
    mm_model = MaxMarginModel(actions, C, N, feature_fn, margin_fn)        
    if constraints:
        mm_model.load_constraints_from_file(constraints)
    return mm_model

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
    max = 1
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

def get_bias_feature_fn(actionfile):
    if type(actionfile) is str:
        actionfile = h5py.File(actionfile, 'r')
    act_set = ActionSet(actionfile)
    return (act_set.bias_features, act_set.num_actions + 1, actionfile)

def get_quad_feature_fn(actionfile):
    if type(actionfile) is str:
        actionfile = h5py.File(actionfile, 'r')
    act_set = ActionSet(actionfile)
    return (act_set.quad_features, 2 + 2*act_set.num_actions, actionfile)

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
    """
    def __init__(self, actionfile, use_cache = True):
        self.actionfile = actionfile
        self.actions = sorted(actionfile.keys())
        self.action_to_ind = dict((v, i) for i, v in enumerate(self.actions))
        self.num_actions = len(self.actions)
        self.action_margin_cache = {}
        self.action_state_cache = {}
        self.use_cache = use_cache
        self.link_names = ["%s_gripper_tool_frame"%lr for lr in ('lr')]

    def get_ds_cloud(self, action):
        return clouds.downsample(self.actionfile[action]['cloud_xyz'], DS_SIZE)
    
    def bias_features(self, state, action):
        feat = np.zeros(self.num_actions + 1)
        feat[0] = registration_cost(state[1], self.get_ds_cloud(action))
        feat[self.action_to_ind[action]+1] = 1
        return feat
    
    def quad_features(self, state, action):
        feat = np.zeros(2 + 2*self.num_actions)
        s = registration_cost(state[1], self.get_ds_cloud(action))
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
        (cache_hit, value) = self.check_am_cache(a1, a2)
        if cache_hit:
            print 'action_only_margin_cache hit'
            return value
        warped_a1_trajs = [warp_hmats(self.get_ds_cloud(a1),
                                      self.get_ds_cloud(a2),
                                      self.actionfile[a1][ln]['hmat'][:]) for ln in self.link_names]
        a1_trajs = [self.actionfile[a1][ln]['hmat'][:] for ln in self.link_names]
        warped_a2_trajs = [warp_hmats(self.get_ds_cloud(a2),
                                      self.get_ds_cloud(a1),
                                      self.actionfile[a2][ln]['hmat'][:]) for ln in self.link_names]
        a2_trajs = [self.actionfile[a2][ln]['hmat'][:] for ln in self.link_names]
        ret_val = sum(compare_hmats(t1, t2) for (t1, t2) in 
                          zip(warped_a1_trajs + warped_a2_trajs, a2_trajs + a1_trajs))
        self.am_cache_store(a1, a2, ret_val)
        return ret_val

    def check_am_cache(self, a1, a2):
        if self.use_cache:
            for key in [(a1, a2), (a2, a1)]:
                if key in self.action_margin_cache:
                    return (True, self.action_margin_cache[key])
        return False, False

    def am_cache_store(self, a1, a2, v):
        if self.use_cache:
            self.action_margin_cache[(a1, a2)] = v

    def check_asm_cache(self, s, a):
        # cache stores the warped traj for a
        if self.use_cache:
            key = (s[0], a)
            if key in self.action_state_cache:
                return (True, self.action_state_cache[key])
        return False, False

    def store_asm_cache(self, s, a, v):
        if self.use_cache:
            key = (s[0], a)
            self.action_state_cache[key] = v

    def clear_asm_cache(self):
        self.action_state_cache = {}

    def action_state_margin(self, state, a1, a2):
        """
        look at the difference for both when warped to state
        ex. w1 is warp(a1, state), w2 is warp(a2, state)
        compare_hmats(w1(a1.traj), w2(a2.traj))

        might be worth it to implement cacheing at state/action level here
        when we call this with a particular expert demo we will warp that trajectory
        once for each action we compare to -- issue is hashing point clouds effectively        
        """
        a1_hit, value = self.check_asm_cache(state, a1)
        if a1_hit:
            warped_a1_trajs = value
        else:
            warped_a1_trajs = [warp_hmats(self.get_ds_cloud(a1),
                                          state[1],
                                          self.actionfile[a1][ln]['hmat']) for ln in self.link_names]
            self.store_asm_cache(state, a1, warped_a1_trajs)
        a2_hit, value = self.check_asm_cache(state, a2)
        if a2_hit:
            warped_a2_trajs = value
        else:
            warped_a2_trajs = [warp_hmats(self.get_ds_cloud(a2),
                                      state[1],
                                      self.actionfile[a2][ln]['hmat']) for ln in self.link_names]
            self.store_asm_cache(state, a2, warped_a2_trajs)
        return sum(compare_hmats(t1, t2) for (t1, t2) in zip(warped_a1_trajs, warped_a2_trajs))

    def combined_margin(self, state, a1, a2):
        return (self.action_only_margin(a1, a2) +
                self.action_state_margin(a1, a2, state))
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

def warp_hmats(xyz_src, xyz_targ, hmats):
    if not use_rapprentice:
        return hmats
    scaled_xyz_src, src_params = registration.unit_boxify(xyz_src)
    scaled_xyz_targ, targ_params = registration.unit_boxify(xyz_targ)        
    f,_ = registration.tps_rpm_bij(scaled_xyz_src, scaled_xyz_targ, plot_cb = None,
                                   plotting=0,rot_reg=np.r_[1e-4,1e-4,1e-1], 
                                   n_iter=50, reg_init=10, reg_final=.1)
    f = registration.unscale_tps(f, src_params, targ_params)
    return f.transform_hmats(hmats)

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('constraintfile')
    parser.add_argument('modelfile')
    parser.add_argument('weightfile', nargs='?', default=None)
    parser.add_argument("--quad_features", action="store_true")
    parser.add_argument('--build_constraints', nargs=1, metavar='demofile')
    args = parser.parse_args()
    if args.quad_features:
        print "using quadratic features"
        (feature_fn, num_features, act_file) = get_quad_feature_fn('data/all.h5')
    else:
        print "using bias features"
        (feature_fn, num_features, act_file) = get_bias_feature_fn('data/all.h5')
    (margin_fn, act_file) = get_action_state_margin_fn(act_file)
    C = 1 # hyperparameter

    if args.build_constraints is not None:
        print 'Building constraints into {}.'.format(args.constraintfile)
        mm_model = rope_max_margin_model(act_file, C, num_features, feature_fn, margin_fn)
        add_constraints_from_demo(mm_model, args.build_constraints[0], outfile=args.constraintfile, verbose=True)
    else:
        if os.path.exists(args.modelfile):
            print 'Found model: {}'.format(args.modelfile)
            model = MaxMarginModel.read(args.modelfile, act_file.keys(), feature_fn, margin_fn)
            model.optimize_model()
        else:
            print 'Building and optimizing model.'
            mm_model = rope_max_margin_model(act_file, C, num_features, feature_fn, margin_fn, \
                                             args.constraintfile)
            mm_model.save_model(args.modelfile)
            mm_model.optimize_model()
