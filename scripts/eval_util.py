# Contains useful functions for evaluating on PR2 rope tying simulation.
# The purpose of this class is to eventually consolidate the various
# instantiations of do_task_eval.py
import argparse
import sim_util
import util
import openravepy, trajoptpy
import h5py, numpy as np
from rapprentice import math_utils as mu

class EvalStats:
    def __init__(self, **kwargs):
        self.found_feasible_action = False
        self.success = False
        self.feasible = True
        self.misgrasp = False
        self.action_elapsed_time = 0
        self.exec_elapsed_time = 0
        for k in kwargs:
            setattr(self, k, kwargs[k])

def get_holdout_items(holdoutfile, task_list, task_file, i_start, i_end):
    tasks = [] if task_list is None else task_list
    if task_file is not None:
        file = open(task_file, 'r')
        for line in file.xreadlines():
            try:
                tasks.append(int(line))
            except:
                print "get_specified_tasks:", line, "is not a valid task"
    if i_start != -1 or i_end != -1:
        if i_end == -1:
            i_end = len(holdoutfile)
        if i_start == -1:
            i_start = 0
        tasks.extend(range(i_start, i_end))
    if not tasks:
        return sorted(holdoutfile.iteritems(), key=lambda item: int(item[0]))
    else:
        return [(unicode(t), holdoutfile[unicode(t)]) for t in tasks]

def add_dict_to_group(group, d):
    for (k,v) in d.iteritems():
        if type(v) == dict:
            add_dict_to_group(group.create_group(k), v)
        elif v is None:
            group[k] = 'None'
        elif type(v) == list and len(v) == 0:
            group[k] = 'empty_list'
        else:
            group[k] = v
    return group

def group_to_dict(group):
    d = {}
    for (k,v) in group.iteritems():
        if isinstance(v, h5py.Group):
            d[k] = group_to_dict(v)
        elif v[()] == 'None':
            d[k] = None
        elif v[()] == 'empty_list':
            d[k] = []
        else:
            d[k] = v[()]
    return d

def add_full_trajs_to_group(full_trajs_g, full_trajs):
    for (i_traj, (traj, dof_inds)) in enumerate(full_trajs):
        full_traj_g = full_trajs_g.create_group(str(i_traj))
        # current version of h5py can't handle empty arrays, so don't save them if they are empty
        if np.all(traj.shape):
            full_traj_g['traj'] = traj
        if len(dof_inds) > 0:
            full_traj_g['dof_inds'] = dof_inds

def group_to_full_trajs(full_trajs_g):
    full_trajs = []
    for i_traj in range(len(full_trajs_g)):
        full_traj_g = full_trajs_g[str(i_traj)]
        if 'traj' in full_traj_g and 'dof_inds' in full_traj_g:
            full_traj = (full_traj_g['traj'][()], list(full_traj_g['dof_inds'][()]))
        else:
            full_traj = (np.empty((0,0)), [])
        full_trajs.append(full_traj)
    return full_trajs

def namespace2dict(args):
    args_dict = vars(args).copy()
    for (k,v) in args_dict.iteritems():
        try:
            args_dict[k] = namespace2dict(v)
        except TypeError:
            continue
    return args_dict

def dict2namespace(args_dict):
    args_dict = args_dict.copy()
    for (k,v) in args_dict.iteritems():
        if type(v) is dict:
            args_dict[k] = dict2namespace(v)
    args = argparse.Namespace(**args_dict)
    return args

def save_results_args(fname, args):
    # if args is already in the results file, make sure that the eval arguments are the same
    if fname is None:
        return
    result_file = h5py.File(fname, 'a')

    args_dict = namespace2dict(args)
    if 'args' in result_file:
        loaded_args_dict = group_to_dict(result_file['args'])
        if 'eval' not in loaded_args_dict:
            raise RuntimeError("The file doesn't have eval arguments")
        if 'eval' not in args_dict:
            raise RuntimeError("The current arguments doesn't have eval arguments")
        if set(loaded_args_dict['eval'].keys()) != set(args_dict['eval'].keys()):
            raise RuntimeError("The arguments of the file and the current arguments have different eval arguments")
        for (k, args_eval_val) in args_dict['eval'].iteritems():
            loaded_args_eval_val = loaded_args_dict['eval'][k]
            if np.any(args_eval_val != loaded_args_eval_val):
                raise RuntimeError("The arguments of the file and the current arguments have different eval arguments: %s, %s"%(loaded_args_eval_val, args_eval_val))
    else:
        add_dict_to_group(result_file.create_group('args'), args_dict)
    result_file.close()

def load_results_args(fname):
    if fname is None:
        raise RuntimeError("Cannot load task results with an unspecified file name")
    result_file = h5py.File(fname, 'r')
    args = dict2namespace(group_to_dict(result_file['args']))
    result_file.close()
    return args

def save_task_results_init(fname, task_index, sim_env, init_rope_nodes):
    if fname is None:
        return
    result_file = h5py.File(fname, 'a')
    task_index = str(task_index)
    if task_index in result_file:
        del result_file[task_index]
    result_file.create_group(task_index)
    init_group = result_file[task_index].create_group('init')
    init_group['trans'], init_group['rots'] = sim_util.get_rope_transforms(sim_env)
    init_group['rope_nodes'] = sim_env.sim.rope.GetControlPoints()
    init_group['init_rope_nodes'] = init_rope_nodes
    result_file.close()

def load_task_results_init(fname, task_index):
    if fname is None:
        raise RuntimeError("Cannot load task results with an unspecified file name")
    result_file = h5py.File(fname, 'r')
    task_index = str(task_index)
    init_group = result_file[task_index]['init']
    trans = init_group['trans'][()]
    rots = init_group['rots'][()]
    rope_nodes = init_group['rope_nodes'][()]
    init_rope_nodes = init_group['init_rope_nodes'][()]
    result_file.close()
    return trans, rots, rope_nodes, init_rope_nodes

def save_task_results_step(fname, task_index, step_index, sim_env, best_root_action, q_values_root, full_trajs, eval_stats, **kwargs):
    if fname is None:
        return
    result_file = h5py.File(fname, 'a')
    task_index = str(task_index)
    step_index = str(step_index)
    assert task_index in result_file, "Must call save_task_results_init() before save_task_results_step()"
    if step_index not in result_file[task_index]:
        step_group = result_file[task_index].create_group(step_index)
    else:
        step_group = result_file[task_index][step_index]
    step_group['trans'], step_group['rots'] = sim_util.get_rope_transforms(sim_env)
    step_group['rope_nodes'] = sim_env.sim.rope.GetControlPoints()
    step_group['best_action'] = str(best_root_action)
    step_group['values'] = q_values_root
    add_full_trajs_to_group(step_group.create_group('full_trajs'), full_trajs)
    add_dict_to_group(step_group.create_group('eval_stats'), vars(eval_stats))
    add_dict_to_group(step_group.create_group('kwargs'), kwargs)
    result_file.close()

def load_task_results_step(fname, task_index, step_index):
    if fname is None:
        raise RuntimeError("Cannot load task results with an unspecified file name")
    result_file = h5py.File(fname, 'r')
    task_index = str(task_index)
    step_index = str(step_index)
    step_group = result_file[task_index][step_index]
    trans = step_group['trans'][()]
    rots = step_group['rots'][()]
    rope_nodes = step_group['rope_nodes'][()]
    best_action = step_group['best_action'][()]
    q_values = step_group['values'][()]
    full_trajs = group_to_full_trajs(step_group['full_trajs'])
    eval_stats = EvalStats(**group_to_dict(step_group['eval_stats']))
    kwargs = group_to_dict(step_group['kwargs'])
    result_file.close()
    return trans, rots, rope_nodes, best_action, q_values, full_trajs, eval_stats, kwargs

def save_task_follow_traj_inputs(fname, sim_env, task_index, step_index, choice_index, miniseg_index, manip_name,
                                 new_hmats, old_traj):
    if fname is None:
        return
    result_file = h5py.File(fname, 'a')
    task_index = str(task_index)
    step_index = str(step_index)
    choice_index = str(choice_index)
    miniseg_index = str(miniseg_index)
    assert task_index in result_file, "Must call save_task_results_int() before save_task_follow_traj_inputs()"

    if step_index not in result_file[task_index]:
        result_file[task_index].create_group(step_index)

    if 'plan_traj' not in result_file[task_index][step_index]:
        result_file[task_index][step_index].create_group('plan_traj')
    if choice_index not in result_file[task_index][step_index]['plan_traj']:
        result_file[task_index][step_index]['plan_traj'].create_group(choice_index)
    if miniseg_index not in result_file[task_index][step_index]['plan_traj'][choice_index]:
        result_file[task_index][step_index]['plan_traj'][choice_index].create_group(miniseg_index)
    manip_g = result_file[task_index][step_index]['plan_traj'][choice_index][miniseg_index].create_group(manip_name)

    manip_g['rope_nodes'] = sim_env.sim.rope.GetControlPoints()
    trans, rots = sim_util.get_rope_transforms(sim_env)
    manip_g['trans'] = trans
    manip_g['rots'] = rots
    manip_g['dof_inds'] = sim_env.robot.GetActiveDOFIndices()
    manip_g['dof_vals'] = sim_env.robot.GetDOFValues()
    manip_g.create_group('new_hmats')
    for (i_hmat, new_hmat) in enumerate(new_hmats):
        manip_g['new_hmats'][str(i_hmat)] = new_hmat
    manip_g['old_traj'] = old_traj
    result_file.close()

def save_task_follow_traj_output(fname, task_index, step_index, choice_index, miniseg_index, manip_name, new_joint_traj):
    if fname is None:
        return
    result_file = h5py.File(fname, 'a')
    task_index = str(task_index)
    step_index = str(step_index)
    choice_index = str(choice_index)
    miniseg_index = str(miniseg_index)

    assert task_index in result_file, "Must call save_task_follow_traj_inputs() before save_task_follow_traj_output()"
    assert step_index in result_file[task_index]
    assert 'plan_traj' in result_file[task_index][step_index]
    assert choice_index in result_file[task_index][step_index]['plan_traj']
    assert miniseg_index in result_file[task_index][step_index]['plan_traj'][choice_index]
    assert manip_name in result_file[task_index][step_index]['plan_traj'][choice_index][miniseg_index]

    result_file[task_index][step_index]['plan_traj'][choice_index][miniseg_index][manip_name]['output_traj'] = new_joint_traj
    result_file.close()

def traj_collisions(sim_env, full_traj, collision_dist_threshold, upsample=0):
    """
    Returns the set of collisions. 
    manip = Manipulator or list of indices
    """
    traj, dof_inds = full_traj
    sim_util.unwrap_in_place(traj, dof_inds=dof_inds)

    if upsample > 0:
        traj_up = mu.interp2d(np.linspace(0,1,upsample), np.linspace(0,1,len(traj)), traj)
    else:
        traj_up = traj
    cc = trajoptpy.GetCollisionChecker(sim_env.env)

    with openravepy.RobotStateSaver(sim_env.robot):
        sim_env.robot.SetActiveDOFs(dof_inds)
    
        col_times = []
        for (i,row) in enumerate(traj_up):
            sim_env.robot.SetActiveDOFValues(row)
            col_now = cc.BodyVsAll(sim_env.robot)
            #with util.suppress_stdout():
            #    col_now2 = cc.PlotCollisionGeometry()
            col_now = [cn for cn in col_now if cn.GetDistance() < collision_dist_threshold]
            if col_now:
                #print [cn.GetDistance() for cn in col_now]
                col_times.append(i)
                #print "trajopt.CollisionChecker: ", len(col_now)
            #print col_now2
        
    return col_times

def traj_is_safe(sim_env, full_traj, collision_dist_threshold, upsample=0):
    return traj_collisions(sim_env, full_traj, collision_dist_threshold, upsample) == []
