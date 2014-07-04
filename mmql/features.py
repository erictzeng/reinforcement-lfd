"""
Functions and classes for computing features
"""
import h5py

import numpy as np

from tpsopt.batchtps import SrcContext, TgtContext, batch_tps_rpm_bij
from tpsopt.transformations import unit_boxify

import IPython as ipy
from pdb import pm, set_trace


class Feature(object):
    """
    base class for computing features
    """
    def __init__(self, actionfile):
        raise NotImplementedError

    def feature(self, state, segname):
        """
        returns the feature for this state/segname
        """
        feats = self.features(state)
        ind = self.get_ind(segname)
        return feats[ind]

    def features(self, state):
        """
        returns a dictionary mapping segnames to features
        """
        raise NotImplementedError

    def select_best(self, state, k = 1):
        scores = zip(self.src_ctx.seg_names, np.dot(self.features(state), self.weights))
        scores = sorted(scores, key=lambda x: -x[1]) # we want to max
        return ([x[0] for x in scores[:k]], [x[1] for x in scores[:k]])

    def load_weights(self, fname):
        f = h5py.File(fname, 'r')
        weights = f['weights'][:]
        f.close()
        assert weights.shape == self.weights.shape
        self.weights = weights

    def get_ind(self, a):
        raise NotImplementedError



class BatchRCFeats(Feature):        
    
    def __init__(self, actionfile):
        self.src_ctx = SrcContext()
        self.src_ctx.read_h5(actionfile)
        self.tgt_cld = None
        self.tgt_ctx = TgtContext(self.src_ctx)
        self.name2ind = dict([(s, i) for i, s in enumerate(self.src_ctx.seg_names)])
        self.costs = np.zeros(self.src_ctx.N)
        self.N = len(self.src_ctx.seg_names)
        self.indicators = np.eye(self.N)
        self.weights = np.r_[-1, np.zeros(self.N)]

    def feature(self, state, segname):
        self.tgt_cld = state[1]
        self.tgt_ctx.set_cld(self.tgt_cld)
        self.costs = batch_tps_rpm_bij(self.src_ctx, self.tgt_ctx)
        ind = self.name2ind[segname]
        return np.c_[self.costs[ind], self.indicators[ind]]

    def features(self, state):
        self.tgt_cld = state[1]
        self.tgt_ctx.set_cld(self.tgt_cld)
        self.costs = batch_tps_rpm_bij(self.src_ctx, self.tgt_ctx)
        return np.c_[self.costs, self.indicators]

    def get_ind(self, a):
        return self.name2ind[a]

    @staticmethod
    def get_size(num_actions):
        return num_actions + 1
