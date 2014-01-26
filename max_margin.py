#!/usr/bin/env python

try:
    import gurobipy as grb
    USE_GUROBI = True
    GRB = grb.GRB # constants for gurobi
except ImportError:
    USE_GUROBI = False
import IPython as ipy
import numpy as np
import h5py, random, math, util
from numbers import Number
from pdb import pm, set_trace
import sys, os
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
        self._C = C
        self.w = np.asarray([self.model.addVar(lb = -1*GRB.INFINITY, name = str(i)) 
                             for i in range(N)])
        self.weights = np.zeros(N)
        self.xi = self.model.addVar(lb = 0, name = 'xi')
        self.xi_val = None
        self.model.update()
        # w'*w + C*xi
        self.model.setObjective(np.dot(self.w, self.w) + C*self.xi)
        self.model.update()
        self.feature_fn = feature_fn
        self.margin_fn = margin_fn
        self.constraints_cache = set()

    @staticmethod
    def read_helper(mm_model, fname, actions, feature_fn, margin_fn):
        mm_model.actions = actions[:]
        grb_model = grb.read(fname)
        mm_model.model = grb_model
        w = []
        for var in mm_model.model.getVars():
            try:
                int(var.VarName)
                w.append(var)
            except ValueError:
                pass
        mm_model.N = len(w)
        mm_model.w = np.asarray(w)
        mm_model.populate_slacks()
        mm_model.weights = np.zeros(len(w))
        mm_model.feature_fn = feature_fn
        mm_model.margin_fn = margin_fn
        mm_model.constraints_cache = set()

    @staticmethod
    def read(fname, actions, feature_fn, margin_fn):
        mm_model = MaxMarginModel.__new__(MaxMarginModel)
        MaxMarginModel.read_helper(mm_model, fname, actions, feature_fn, margin_fn)
        assert len(mm_model.model.getVars()) == len(mm_model.xi) + len(mm_model.w), "Number of Gurobi vars mismatches the MaxMarginModel vars"
        return mm_model

    def populate_slacks(self):
        # makes sure that the model being read is a single slack one
        xis = [var for var in self.model.getVars() if var.VarName.startswith('xi')]
        assert len(xis) == 1, "There should only be a single xi in single slack MaxMarginModel"
        self.xi = xis[0]
        self.xi_val = None

    @property
    def C(self):
        return self._C
    @C.setter
    def C(self, value):
        self._C = value
        self.xi.Obj = value
        self.model.update()

    def feature(self, s, a):
        return self.feature_fn(s, a)

    def margin(self, s, a1, a2):
        return self.margin_fn(s, a1, a2)

    def add_constraint(self, expert_action_phi, rhs_action_phi, margin_value, xi_name, update=True):
        """
        function to add a constraint to the model with pre-computed
        features and margins
        pass in xi_name so that the constraints are compatible with multi slack
        """
        # make sure the feature size is consistent with phi
        assert self.N == expert_action_phi.shape[0], "failed adding constraint: size of expert_action_phi is inconsistent with feature size"
        assert self.N == rhs_action_phi.shape[0], "failed adding constraint: size of rhs_action_phi is inconsistent with feature size"

        lhs_coeffs = [(p, w) for p, w in zip(expert_action_phi, self.w) if abs(p) >= eps]
        if not lhs_coeffs:
            lhs = 0
        else:
            lhs = grb.LinExpr(lhs_coeffs)
        rhs_coeffs = [(p, w) for w, p in zip(self.w, rhs_action_phi) if abs(p) >= eps]
        rhs_coeffs.append((-1, self.xi))
        rhs = grb.LinExpr(rhs_coeffs)
        rhs += margin_value
        self.model.addConstr(lhs >= rhs)
        #store the constraint so we can store them to a file later
        self.constraints_cache.add(util.tuplify((expert_action_phi, rhs_action_phi, margin_value, xi_name)))
        if update:
            self.model.update()

    def add_example(self, state, expert_action, xi_name = None, verbose = False):
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
            self.add_constraint(expert_action_phi, rhs_action_phi, margin, xi_name, update=False)
            if verbose:
                print "added {}/{}".format(i, len(self.actions))
        self.model.update()

    def save_constraints_to_file(self, fname, save_weights=False):
        raise NotImplementedError('deprecated')
        outfile = h5py.File(fname, 'w')
        for i, (exp_phi, rhs_phi, margin, xi_name) in enumerate(self.constraints_cache):
            g = outfile.create_group(str(i))
            g['exp_features'] = exp_phi
            g['rhs_phi'] = rhs_phi
            g['margin'] = margin
            if xi_name:
                g['xi'] = xi_name
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
            if 'xi' in constr.keys():
                xi_name = constr['xi'][()]
            else:
                xi_name = None
            self.add_constraint(exp_phi, rhs_phi, margin, xi_name, update=False)
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

    @staticmethod
    def read(fname, actions, feature_fn, margin_fn):
        mm_model = MultiSlackMaxMarginModel.__new__(MultiSlackMaxMarginModel)
        MaxMarginModel.read_helper(mm_model, fname, actions, feature_fn, margin_fn)
        assert len(mm_model.model.getVars()) == len(mm_model.xi) + len(mm_model.w), "Number of Gurobi vars mismatches the MultiSlackMaxMarginModel vars"
        return mm_model

    def populate_slacks(self):
        self.xi = [var for var in self.model.getVars() if var.VarName.startswith('xi')]
        self.xi_val = []

    @property
    def C(self):
        return self._C

    @C.setter
    def C(self, value):
        self._C = value
        for xi_var in self.xi:
            xi_var.Obj = value
        self.model.update()

    def add_constraint(self, expert_action_phi, rhs_action_phi, margin_value, xi_var, update=True):
        """
        function to add a constraint to the model with pre-computed
        features and margins
        """
        lhs_coeffs = [(p, w) for p, w in zip(expert_action_phi, self.w) if abs(p) >= eps]
        if not lhs_coeffs:
            lhs = 0
        else:
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
        
    def add_example(self, state, expert_action, xi_name = None, verbose = False):
        """
        add the constraint that this action be preferred to all other actions in the action set
        to the optimization problem
        """
        expert_action_phi = self.feature(state, expert_action)
        cur_slack = self.add_xi(xi_name)
        for (i, other_a) in enumerate(self.actions):
            if other_a == expert_action:
                continue
            # TODO: compute this for loop in parallel
            rhs_action_phi = self.feature(state, other_a)
            margin = self.margin(state, expert_action, other_a)
            self.add_constraint(expert_action_phi, rhs_action_phi, margin, cur_slack, update=False)
            if verbose:
                print "added {}/{}".format(i, len(self.actions)), cur_slack.VarName
        self.model.update()

    def save_constraints_to_file(self, fname, save_weights=False):
        raise NotImplementedError('deprecated')
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

    @staticmethod
    def update_constraints_file(fname):
        def get_abbr_tuple(exp_features):
            return (exp_features[0], exp_features[1:].tolist().index(1))
        infile = h5py.File(fname, 'r+')
        valiter = infile.itervalues()
        test_group = valiter.next()
        if 'xi' in test_group:
            infile.close()
            return
        xi_names = {}
        for key, constr in infile.iteritems():
            exp_tuple = get_abbr_tuple(constr['exp_features'][:])
            if exp_tuple not in xi_names:
                xi_names[exp_tuple] = 'xi_{}'.format(len(xi_names))
            constr['xi'] = str(xi_names[exp_tuple])
        infile.close()
        
    def load_constraints_from_file(self, fname):
        """
        loads the contraints from the file indicated and adds them to the optimization problem
        """
        MultiSlackMaxMarginModel.update_constraints_file(fname)
        infile = h5py.File(fname, 'r')
        n_other_keys = 0
        if 'weights' in infile:
            self.weights = infile['weights'][:]
            n_other_keys += 1
        if 'xi' in infile:
            self.xi_val = infile['xi'][:]
            n_other_keys += 1
        slack_names = {}
        for key_i in range(len(infile) - n_other_keys):
            constr = infile[str(key_i)]
            exp_phi = constr['exp_features'][:]
            rhs_phi = constr['rhs_phi'][:]
            margin = float(constr['margin'][()])
            xi_name = constr['xi'][()]
            if xi_name not in slack_names:
                xi_var = self.add_xi(xi_name)
                slack_names[xi_name] = xi_var
            self.add_constraint(exp_phi, rhs_phi, margin, slack_names[xi_name], update=False)
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


class BellmanMaxMarginModel(MultiSlackMaxMarginModel):    
    
    def __init__(self, actions, C, D, F, gamma, N, feature_fn, margin_fn, E=100):
        MultiSlackMaxMarginModel.__init__(self, actions, C, N, feature_fn, margin_fn)
        self.action_reward = -1
        self.goal_reward = 10
        self._D = D
        self.yi = []
        self.yi_val = []
        self.gamma = gamma
        self._E = E
        self.zi = []
        self.zi_val = []
        self._F = F # weight on the sum of value fns
        self.F_no_norm = F # keeps track of what we want the coefficient on the whole sum to be
        self.f_sum_size = 0

    @staticmethod
    def read(fname, actions, feature_fn, margin_fn):
        mm_model = BellmanMaxMarginModel.__new__(BellmanMaxMarginModel)
        MaxMarginModel.read_helper(mm_model, fname, actions, feature_fn, margin_fn)
        assert len(mm_model.model.getVars()) == len(mm_model.xi) + len(mm_model.yi)+ len(mm_model.w), "Number of Gurobi vars mismatches the BellmanMaxMarginModel vars"
        param_fname = mm_model.get_param_fname(fname)
        param_f = h5py.File(param_fname, 'r')
        mm_model.action_reward = param_f['action_reward'][()]
        mm_model.goal_reward = 10
        mm_model.gamma = param_f['gamma'][()]
        mm_model.f_sum_size = param_f['f_sum_size'][()]
        mm_model.F_no_norm = param_f['F_no_norm'][()]
        mm_model._F = mm_model.F_no_norm/float(mm_model.f_sum_size)
        return mm_model

    def populate_slacks(self):
        self.xi = [var for var in self.model.getVars() if var.VarName.startswith('xi')]
        self.xi_val = []
        self.yi = [var for var in self.model.getVars() if var.VarName.startswith('yi')]
        self.yi_val = []
        
    @property
    def D(self):
        return self._D

    @D.setter
    def D(self, value):
        self._D = value
        for yi_var in self.yi:
            yi_var.Obj = value
        self.model.update()


    @property
    def F(self):
        return self._F

    @F.setter
    def F(self, value):
        for w in self.w:
            w.Obj = w.Obj/float(self._F)
        self.model.update()
        self._F = value/float(self.f_sum_size)
        for w in self.w:
            w.Obj *= float(self._F)
        self.model.update()

    @property
    def E(self):
        return self._E

    @E.setter
    def E(self, value):
        self._E = value
        for zi_var in self.zi:
            zi_var.Obj = value
        self.model.update()
    
    def add_bellman_constraint(self, curr_action_phi, next_action_phi, yi_var, update=True):
        lhs_coeffs = [(p, w) for w, p in zip(self.w, curr_action_phi) if abs(p) >= eps]
        if not lhs_coeffs:
            lhs = 0
        else:
            lhs = grb.LinExpr(lhs_coeffs)
        rhs_coeffs = [(self.gamma*p, w) for w, p in zip(self.w, next_action_phi) if abs(p) >= eps]
        rhs_coeffs.append((1, yi_var)) #flip
        rhs = grb.LinExpr(rhs_coeffs)
        rhs += self.action_reward
        # w'*curr_phi <= -1 + yi + gammma * w'*next_phi
        self.model.addConstr(lhs <= rhs) #flip
        #store the constraint so we can store them to a file later
        self.constraints_cache.add(util.tuplify((curr_action_phi, next_action_phi, 0, yi_var.VarName)))
        if update:
            self.model.update()

    def add_yi(self, yi_name):
        new_yi = self.model.addVar(lb = 0, name = yi_name, obj = self.D)
        # make sure new_yi is not already in self.yi
        assert len([yi for yi in self.yi if yi is new_yi]) == 0
        self.yi.append(new_yi)
        self.model.update()
        return new_yi
    
    # this function doesn't add examples
    def add_trajectory(self, states_actions, yi_name, verbose=False):
        cur_slack = self.add_yi(yi_name)
        features = range(len(states_actions))
        for i in range(len(states_actions)-1):
            state, action = states_actions[i]
            next_state, next_action = states_actions[i+1]
            curr_action_phi = self.feature(state, action)
            next_action_phi = self.feature(next_state, next_action)
            features[i] = curr_action_phi
            features[i+1] = next_action_phi #bestpractices
            self.add_bellman_constraint(curr_action_phi, next_action_phi, cur_slack, update=False)
            if verbose:
                sys.stdout.write("added bellman constraint {}/{} ".format(i, len(states_actions)-1) + str(cur_slack.VarName) + '\r')
                sys.stdout.flush()
        for feat in features:
            for (i, w) in enumerate(self.w):
                if abs(feat[i]) >= eps:
                    w.Obj -= self.F*feat[i] #flip
            self.f_sum_size += 1
        self.model.update()
        self.F = self.F_no_norm        # this will update the coeffiencts to take into account num_values


    def add_goal_constraint(self, prev_state, prev_action, update=True):
        """
        Adds constraints specifying w'*phi = -1 + gamma * goal_reward + zi
        """
        prev_action_phi = self.feature(prev_state, prev_action)
        lhs_coeffs = [(p, w) for w, p in zip(self.w, prev_action_phi) if abs(p) >= eps]
        if not lhs_coeffs:
            lhs = 0
        else:
            lhs = grb.LinExpr(lhs_coeffs)
        rhs = self.action_reward + self.gamma * self.goal_reward
        self.model.addConstr(lhs <= rhs) #flip
        if update:
            self.model.update()

    def add_goal_constraints(self, fname):
        """
        Adds constraints specifying w'*phi <= -1 + gamma * goal_reward
        fname must specify a file of labelled examples.
        NOTE: We assume the examples in fname have integer ids in consecutive order, starting from 0
        """
        demofile = h5py.File(fname, 'r')
        for k in range(len(f.keys())):
            if f[str(k)]['knot'][()] == 1:
                prev_state = f[str(k-1)]['cloud_xyz']
                prev_action = f[str(k-1)]['action']
                self.add_goal_constraint(prev_state, prev_action, update=False)
        self.model.update()

    def load_constraints_from_file(self, fname):
        """
        loads the contraints from the file indicated and adds them to the optimization problem
        """
        MultiSlackMaxMarginModel.update_constraints_file(fname)
        infile = h5py.File(fname, 'r')
        n_other_keys = 0
        if 'weights' in infile:
            self.weights = infile['weights'][:]
            n_other_keys += 1
        if 'xi' in infile:
            self.xi_val = infile['xi'][:]
            n_other_keys += 1
        xi_names = {}
        yi_names = {}
        action_phis = {}
        for key_i in range(len(infile) - n_other_keys):
            constr = infile[str(key_i)]
            slack_name = constr['xi'][()]
            if slack_name.startswith('yi'):
                curr_action_phi = constr['exp_features'][:]
                next_action_phi = constr['rhs_phi'][:]
                example = constr['example'][()]
                if slack_name not in yi_names:
                    yi_var = self.add_yi(slack_name)
                    yi_names[slack_name] = yi_var
                self.add_bellman_constraint(curr_action_phi, next_action_phi, yi_names[slack_name], update=False)
                traj_i = slack_name[3:]
                curr_state_i, next_state_i, _ = example.split('-')
                if traj_i not in action_phis:
                    action_phis[traj_i] = {}
                action_phis[traj_i][curr_state_i] = curr_action_phi
                action_phis[traj_i][next_state_i] = next_action_phi
            else:
                exp_phi = constr['exp_features'][:]
                rhs_phi = constr['rhs_phi'][:]
                margin = float(constr['margin'][()])
                if slack_name not in xi_names:
                    xi_var = self.add_xi(slack_name)
                    xi_names[slack_name] = xi_var
                self.add_constraint(exp_phi, rhs_phi, margin, xi_names[slack_name], update=False)
        infile.close()
        # add to the objective the values that are in the bellman constraint
        for features in action_phis.values():
            assert len(features) > 2, "Some trajectories has less than 3 steps. Did you fix the the constraints file?"
            for feat in features.values():
                for (i, w) in enumerate(self.w):
                    if abs(feat[i]) >= eps:
                        w.Obj -= self.F*feat[i] #flip
                self.f_sum_size += 1
        self.F = self.F_no_norm        # this will update the coeffiencts to take into account num_values
        self.model.update()
        
    def load_weights_from_file(self, fname):
        infile = h5py.File(fname, 'r')
        self.weights = infile['weights'][:]
        if 'xi' in infile:
            self.xi_val = infile['xi'][()]
        if 'yi' in infile:
            self.yi_val = infile['yi'][()]
        infile.close()
        
    def save_weights_to_file(self, fname):
        # changed to use h5py.File so file i/o is consistent
        outfile = h5py.File(fname, 'w')
        outfile['weights'] = self.weights
        outfile['xi'] = self.xi_val
        outfile['yi'] = self.yi_val
        outfile.close()

    def get_param_fname(self, fname):
        fname_noext = os.path.splitext(fname)[0]
        return fname_noext + '_param.h5'                

    def save_model(self, fname):
        MaxMarginModel.save_model(self, fname)
        param_fname = self.get_param_fname(fname)
        param_f = h5py.File(param_fname, 'w')
        param_f['gamma'] = self.gamma
        param_f['action_reward'] = self.action_reward
        param_f['F_no_norm'] = self.F_no_norm
        param_f['f_sum_size'] = self.f_sum_size
        
    def optimize_model(self):
        self.model.update()
        self.model.optimize()
        try:
            self.weights = [x.X for x in self.w]
            self.xi_val = [x.X for x in self.xi]
            self.yi_val = [x.X for x in self.yi]
            return self.weights
        except grb.GurobiError:
            raise RuntimeError, "issue with optimizing model, check gurobi optimizer output"

def grid_test_fns():
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
    def sample_state():
        return np.asarray([random.random()*10 for i in range(2)])
    return (actions, C, N, feat, margin, select_action, sample_state)

def gen_labels(sample_state, select_action, N = 100):
    exp_demos = []
    for j in range(N):
        s = sample_state()
        a = select_action(s)
        exp_demos.append((s, a))
    return exp_demos

def test_model(model_class):
    (actions, C, N, feat, margin, select_action, sample_state) = grid_test_fns()

    expert_demos = gen_labels(sample_state, select_action)
    mm = model_class(actions, C, N, feat, margin)

    for (s, a) in expert_demos:
        mm.add_example(s, a)

    weights = mm.optimize_model()

    expert_actions = []
    learned_actions = []
    for j in range(100):
        s = sample_state()
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

def test_update_constraints():
    (actions, C, N, feat, margin, select_action, sample_state) = grid_test_fns()

    expert_demos = gen_labels(sample_state, select_action)
    mm_orig = MaxMarginModel(actions, C, N, feat, margin)
    for (s, a) in expert_demos:
        mm_orig.add_example(s, a)
    mm_orig.save_constraints_to_file('test.h5')
    MultiSlackMaxMarginModel.update_constraints_file('test.h5')
    msmm_loaded = MultiSlackMaxMarginModel(actions, C, N, feat, margin)
    msmm_loaded.load_constraints_from_file('test.h5')

    msmm_orig = MultiSlackMaxMarginModel(actions, C, N, feat, margin)
    for (s, a) in expert_demos:
        msmm_orig.add_example(s, a)

    msmm_loaded.optimize_model()
    msmm_orig.optimize_model()
    return not any(w1 - w2 > eps for w1, w2 in zip(msmm_loaded.weights, msmm_orig.weights))

def test_bellman():
    grid_dim=50
    goal_state = np.array([grid_dim,grid_dim])
    actions = {"n":np.array([0,1]), "e":np.array([1,0]), "s":np.array([0,-1]), "w":np.array([-1,0])}
    def feature_fn(state, action):
        action = actions[action]
        next_state = state+action
        diff = goal_state - next_state
        return np.r_[np.abs(diff), np.linalg.norm(diff)]
    def gen_trajectory():
        traj = []
        state = np.array([random.randint(0,grid_dim), random.randint(0,grid_dim)])
        while all(state!=goal_state):
            diff = goal_state - state
            action = np.zeros(2)
            action[np.argmax(np.abs(diff))] = 1
            action = np.sign(diff)*action
            for k in actions:
                if np.linalg.norm(actions[k] - action) < eps:
                    traj.append((state, k))
            state = state + action
        return traj
    def margin_fn(state, action1, action2):
        action1 = actions[action1]
        action2 = actions[action2]
        state1 = state + action1
        state2 = state + action2
        d1 = np.linalg.norm(goal_state - state1, 1)
        d2 = np.linalg.norm(goal_state - state2, 1)
        return abs(d1-d2)
    C = 2
    D = 1
    gamma = 0.9
    N = 3
    model = BellmanMaxMarginModel(actions.keys(), C, D, gamma, N, feature_fn, margin_fn)
    for i in range(50):
        traj = gen_trajectory()
        for state, action in traj:
            model.add_example(state, action)
        model.add_trajectory(traj, 'yi%i'%i)
        if len(traj) > 0:
            prev_state = traj[-1][0]
            prev_action = traj[-1][1]
            model.add_goal_constraint(prev_state, prev_action)
    weights = model.optimize_model()
    print weights 
    return True

if __name__ == '__main__':
    print 'Testing out bellman model'
    if not test_bellman(): print 'Unit Test Failed'
    
    print 'Testing out single slack max margin model'
    if not test_model(MaxMarginModel): print 'Unit Test Failed'

    print 'Testing out multi-slack max margin model'
    if not test_model(MultiSlackMaxMarginModel): print 'Unit Test Failed'

    print 'Testing update constraints'
    if not test_update_constraints(): print 'Unit Test Failed'
