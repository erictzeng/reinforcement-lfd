#!/usr/bin/env python

import gurobipy as grb
import IPython as ipy
import numpy as np
import h5py, random, math, util
from pdb import pm
GRB = grb.GRB # constants for gurobi
eps = 10**-8
MAX_ITER=1000

"""
functions  and classes for setting up and running max margin, uses gurobi as the basic optimizer
"""



class MaxMarginModel(object):
    
    def __init__(self, actions, C, N, feature_fn, margin_fn):
        """
        @param actions: list of the actions that we're going to be deciding between
        @param C: value of the hyper-parameter for the optiimization
        @param N: number of features for each state/action pair, controls the number of 
                  variables in the optimization
        @param feature_fn: function that maps (action, state) pairs to features
        @param margin_fn: function that maps (action, action, state) to margin b/t actions
        """
        self.model = grb.Model()
        self.actions = actions[:]
        self.N = N
        self.C = C
        self.w = np.asarray([self.model.addVar(lb = -1*GRB.INFINITY, name = str(i)) 
                             for i in range(N)])
        self.weights = np.zeros(N)
        self.xi = self.model.addVar(lb = 0, name = 'xi')
        self.xi_val = None
        self.model.update()
        # w'*w + C*xi
        self.model.setObjective(np.dot(self.w, self.w) + self.C*self.xi)
        self.model.update()
        self.feature_fn = feature_fn
        self.margin_fn = margin_fn
        self.constraints_cache = set()

    def feature(self, s, a):
        return self.feature_fn(s, a)

    def margin(self, s, a1, a2):
        return self.margin_fn(s, a1, a2)

    def add_constraint(self, expert_action_phi, rhs_action_phi, margin_value, update=True):
        """
        function to add a constraint to the model with pre-computed
        features and margins
        """
        # make sure the feature size is consistent with phi
        assert self.N == expert_action_phi.shape[0], "failed adding constraint: size of expert_action_phi is inconsistent with feature size"
        assert self.N == rhs_action_phi.shape[0], "failed adding constraint: size of rhs_action_phi is inconsistent with feature size"
        
        lhs_coeffs = [(p, w) for p, w in zip(expert_action_phi, self.w) if abs(p) >= eps]
        lhs = grb.LinExpr(lhs_coeffs)
        rhs_coeffs = [(p, w) for w, p in zip(self.w, rhs_action_phi) if abs(p) >= eps]
        rhs_coeffs.append((-1, self.xi))
        rhs = grb.LinExpr(rhs_coeffs)
        rhs += margin_value
        self.model.addConstr(lhs >= rhs)
        #store the constraint so we can store them to a file later
        self.constraints_cache.add(util.tuplify((expert_action_phi, rhs_action_phi, margin_value)))
        if update:
            self.model.update()

    def add_example(self, state, expert_action, verbose = False):
        """
        add the constraint that this action be preferred to all other actions in the action set
        to the optimization problem
        """
        expert_action_phi = self.feature(state, expert_action)
        for (i, other_a) in enumerate(self.actions):
            if other_a == expert_action:
                continue
            # TODO: compute this for loop in parallel
            rhs_action_phi = self.feature(state, other_a)
            margin = self.margin(state, expert_action, other_a)
            self.add_constraint(expert_action_phi, rhs_action_phi, margin, update=False)
            if verbose:
                print "added {}/{}".format(i, len(self.actions))
        self.model.update()

    def save_constraints_to_file(self, fname, save_weights=False):
        outfile = h5py.File(fname, 'w')
        for i, (exp_phi, rhs_phi, margin) in enumerate(self.constraints_cache):
            g = outfile.create_group(str(i))
            g['exp_features'] = exp_phi
            g['rhs_phi'] = rhs_phi
            g['margin'] = margin
        if save_weights:
            outfile['weights'] = self.weights
            outfile['xi'] = self.xi_val
        outfile.close()
        
    def load_constraints_from_file(self, fname):
        """
        loads the contraints from the file indicated and adds them to the optimization problem
        """
        infile = h5py.File(fname, 'r')
        for key, constr in infile.iteritems():
            if key in ('weights', 'xi'): continue
            exp_phi = constr['exp_features'][:]
            rhs_phi = constr['rhs_phi'][:]
            margin = float(constr['margin'][()])
            self.add_constraint(exp_phi, rhs_phi, margin, update=False)
        if 'weights' in infile:
            self.weights = infile['weights'][:]
        if 'xi' in infile:
            self.xi_val = infile['xi'][()]
        infile.close()
        self.model.update()

    def load_weights_from_file(self, fname):
        infile = h5py.File(fname, 'r')
        self.weights = infile['weights'][:]
        if 'xi' in infile:
            self.xi_val = infile['xi'][()]
        infile.close()
        
    def save_weights_to_file(self, fname):
        # changed to use h5py.File so file i/o is consistent
        outfile = h5py.File(fname, 'w')
        outfile['weights'] = self.weights
        outfile['xi'] = self.xi_val
        outfile.close()

    def optimize_model(self):
        self.model.update()
        self.model.optimize()
        try:
            self.weights = [x.X for x in self.w]
            self.xi_val = self.xi.X
            return self.weights
        except grb.GurobiError:
            raise RuntimeError, "issue with optimizing model, check gurobi optimizer output"
    
    def best_action(self, s):
        besti = np.argmax([np.dot(self.w, self.feature(s, a)).getValue() for a in self.actions])
        return (besti, self.actions[besti])

    def save_model(self, fname):
        self.model.write(fname)

class MultiSlackMaxMarginModel(MaxMarginModel):
    
    def __init__(self, actions, C, N, feature_fn, margin_fn):
        """
        @param actions: list of the actions that we're going to be deciding between
        @param C: value of the hyper-parameter for the optiimization
        @param N: number of features for each state/action pair, controls the number of 
                  variables in the optimization
        @param feature_fn: function that maps (action, state) pairs to features
        @param margin_fn: function that maps (action, action, state) to margin b/t actions
        """
        MaxMarginModel.__init__(self, actions, C, N, feature_fn, margin_fn)
        self.model.setObjective(np.dot(self.w, self.w)) # we'll be adding in slacks per constraint
        self.model.remove(self.xi)
        self.xi = []            #  list to keep track of slack variables
        self.xi_val = []

    def add_constraint(self, expert_action_phi, rhs_action_phi, margin_value, xi_var, update=True):
        """
        function to add a constraint to the model with pre-computed
        features and margins
        """
        lhs_coeffs = [(p, w) for p, w in zip(expert_action_phi, self.w) if abs(p) >= eps]
        lhs = grb.LinExpr(lhs_coeffs)
        rhs_coeffs = [(p, w) for w, p in zip(self.w, rhs_action_phi) if abs(p) >= eps]
        rhs_coeffs.append((-1, xi_var))
        rhs = grb.LinExpr(rhs_coeffs)
        rhs += margin_value
        self.model.addConstr(lhs >= rhs)
        #store the constraint so we can store them to a file later
        self.constraints_cache.add(util.tuplify((expert_action_phi, rhs_action_phi, margin_value, xi_var.VarName)))
        if update:
            self.model.update()

    def add_xi(self, xi_name = None):
        if not xi_name:
            xi_name = 'xi_{}'.format(len(self.xi))
        new_xi = self.model.addVar(lb = 0, name = xi_name, obj = self.C)
        self.xi.append(new_xi)
        self.model.update()
        return new_xi
        
    def add_example(self, state, expert_action, verbose = False):
        """
        add the constraint that this action be preferred to all other actions in the action set
        to the optimization problem
        """
        expert_action_phi = self.feature(state, expert_action)
        cur_slack = self.add_xi()
        for (i, other_a) in enumerate(self.actions):
            if other_a == expert_action:
                continue
            # TODO: compute this for loop in parallel
            rhs_action_phi = self.feature(state, other_a)
            margin = self.margin(state, expert_action, other_a)
            self.add_constraint(expert_action_phi, rhs_action_phi, margin, cur_slack, update=False)
            if verbose:
                print "added {}/{}".format(i, len(self.actions))
        self.model.update()

    def save_constraints_to_file(self, fname, save_weights=False):
        outfile = h5py.File(fname, 'w')
        for i, (exp_phi, rhs_phi, margin, xi_name) in enumerate(self.constraints_cache):
            g = outfile.create_group(str(i))
            g['exp_features'] = exp_phi
            g['rhs_phi'] = rhs_phi
            g['margin'] = margin
            g['xi'] = str(xi_name)
        if save_weights:
            outfile['weights'] = self.weights
            outfile['xi'] = self.xi_val
        outfile.close()
        
    def load_constraints_from_file(self, fname):
        """
        loads the contraints from the file indicated and adds them to the optimization problem
        """
        infile = h5py.File(fname, 'r')
        slack_names = {}
        for key, constr in infile.iteritems():
            if key in ('weights', 'xi'): continue
            exp_phi = constr['exp_features'][:]
            rhs_phi = constr['rhs_phi'][:]
            margin = float(constr['margin'][()])
            xi_name = constr['xi'][()]
            if xi_name not in slack_names:
                xi_var = self.add_xi(xi_name)
                slack_names[xi_name] = xi_var
            self.add_constraint(exp_phi, rhs_phi, margin, slack_names[xi_name], update=False)
        if 'weights' in infile:
            self.weights = infile['weights'][:]
        if 'xi' in infile:
            self.xi_val = infile['xi'][:]
        infile.close()
        self.model.update()

    def load_weights_from_file(self, fname):
        infile = h5py.File(fname, 'r')
        self.weights = infile['weights'][:]
        if 'xi' in infile:
            self.xi_val = infile['xi'][:]
        infile.close()

    def optimize_model(self):
        self.model.update()     # this might not be necessary, but w/e
        self.model.optimize()
        try:
            self.weights = [x.X for x in self.w]
            self.xi_val = [x.X for x in self.xi]
            return self.weights
        except grb.GurobiError:
            raise RuntimeError, "issue with optimizing model, check gurobi optimizer output"

def test_model(model_class):
    """
    Test Example: 2d grid -- 
    3 Actions: 
         0: [5, 5]
         1: [10, 10]
         2: [0, 0]
    The right action to take is the closest one
    """
    actions = [(5, 5), (10, 10), (0, 0)]
    N = 4
    C = .1
    def feat(s, a):
        #our features are the real thing plus a bias term
        f = np.zeros(N)
        f[0] = np.linalg.norm(s - np.asarray(a))
        f[1 + actions.index(a)] = 1
        return f

    # TODO: add in structured margin and 
    # introduce some expert suboptimality
    def margin(s, a1, a2):
        return 1
    def select_action(s):
        vals = [np.linalg.norm(s - a) for a in actions]
        return actions[np.argmin(vals)]

    expert_demos = []
    for j in range(100):
        s = np.asarray([random.random()*10 for i in range(2)])
        a = select_action(s)
        expert_demos.append((s, a))

    mm = model_class(actions, C, N, feat, margin)

    for (s, a) in expert_demos:
        mm.add_example(s, a)

    weights = mm.optimize_model()

    expert_actions = []
    learned_actions = []
    for j in range(100):
        s = np.asarray([random.random()*10 for i in range(2)])
        expert_actions.append(select_action(s))
        learned_actions.append(mm.best_action(s)[1])

    mm.save_constraints_to_file('test.h5', save_weights=True)
    mm_2 = model_class(actions, C, N, feat, margin)
    mm_2.load_constraints_from_file('test.h5')
    loaded_weights = mm_2.weights
    loaded_xi = mm_2.xi_val
    weights_2 = mm_2.optimize_model()

    # storing constraints and recomputing max margin shouldn't change result by much
    load_weight_diff = np.linalg.norm(np.asarray(weights) - np.asarray(loaded_weights))
    load_xi_diff = np.linalg.norm(np.asarray(loaded_xi) - np.asarray(mm.xi_val))
    recomputed_weight_diff = np.linalg.norm(np.asarray(weights) - np.asarray(weights_2))
    print 'load weights difference', load_weight_diff
    print 'load xi computed difference', load_xi_diff
    print 're-computed weight difference', recomputed_weight_diff
    # error rate should be low because we have the value we're minimizing as a feature
    err_rate = np.mean(np.asarray(expert_actions) != np.asarray(learned_actions))
    print "error rate", err_rate
    # so we can see what the weights are
    print 'weights', weights
    # the amount of slack
    print 'xi', mm.xi_val
    errs = [x > eps for x in [load_weight_diff, load_xi_diff, recomputed_weight_diff]]
    return not any(errs)
    

if __name__ == '__main__':
    print 'Testing out single slack max margin model'
    if not test_model(MaxMarginModel): print 'Unit Test Failed'

    print 'Testing out multi-slack max margin model'
    if not test_model(MultiSlackMaxMarginModel): print 'Unit Test Failed'
    
    
    
                    
