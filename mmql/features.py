"""
Functions and classes for computing features
"""
import h5py

import tpsopt
from tpsopt.batchtps import SrcContext, TgtContext, batch_tps_rpm_bij
from tpsopt.transformations import unit_boxify
from defaults import *

from rapprentice import ropesim

import IPython as ipy
from pdb import pm, set_trace


class Feature(object):
    """
    base class for computing features
    """
    def __init__(self, actionfile, feature_fn):
        raise NotImplementedError

    def feature(self, state, segname):
        """
        returns the feature for this state/segname
        """
        raise NotImplementedError

    def all_features(self, state):
        """
        returns a dictionary mapping segnames to features
        """
        raise NotImplementedError

    def select_best(self, state, comparator):
        """
        returns the segname of the best action
        """
        raise NotImplementedError


class BatchRCFeats(Feature):
    def __init__(self, actionfile):
        self.src_ctx = SrcContext()
        self.src_ctx.read_h5(actionfile)
        self.tgt_cld = None
        self.tgt_ctx = TgtContext(self.src_ctx)
        self.name_to_ind = dict([(s, i) for i, s in enumerate(self.src_ctx.seg_names)])
        self.costs = np.zeros(self.src_ctx.N)

    def feature(self, state, segname):
        self.tgt_cld = state[1]
        self.tgt_ctx.set_cld(self.tgt_cld)
        self.costs = batch_tps_rpm_bij(self.src_ctx, self.tgt_ctx)
        return self.costs[self.name_to_ind[segname]]

    def all_features(self, state):
        self.tgt_cld = state[1]
        self.tgt_ctx.set_cld(unit_boxify(self.tgt_cld)[0])
        self.costs = batch_tps_rpm_bij(self.src_ctx, self.tgt_ctx)
        return dict(zip(self.src_ctx.seg_names, self.costs))

    def select_best(self, state, k = 1):
        scores = sorted(self.all_features(state).items(), key=lambda x: x[1])
        return ([x[0] for x in scores[:k]], [-1 * x[1] for x in scores[:k]])        

"""
vestigial feature functions from do_task_eval
"""


def regcost_feature_fn(sim_env, state, action, args_eval):
    f, corr = register_tps_cheap(sim_env, state, action, args_eval)
    if f is None:
        cost = np.inf
    else:
        cost = registration.tps_reg_cost(f)
    return np.array([float(cost)]) # no need to normalize since bending cost is independent of number of points

def regcost_trajopt_feature_fn(sim_env, state, action, args_eval):
    obj_values_sum = compute_trans_traj(sim_env, state, action, None, args_eval, simulate=False, transferopt='finger')
    return np.array([obj_values_sum])

def jointopt_feature_fn(sim_env, state, action, args_eval):
    obj_values_sum = compute_trans_traj(sim_env, state, action, None, args_eval, simulate=False, transferopt='joint')
    return np.array([obj_values_sum])

def q_value_fn(state, action, sim_env, fn):
    return np.dot(WEIGHTS, fn(sim_env, state, action)) #+ w0
