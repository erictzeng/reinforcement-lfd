#!/usr/bin/env python

import pprint
import argparse
usage="""
To label data
./do_task_expert_label.py /home/dhm/sampledata/overhand/all.h5 --fake_data_segment=demo1-seg00 --animation=1 --simulation=1 --select_manual --results_outfile=expert_demos.h5

Use results_outfile=expert_demos_test.h5 to explore and get used to interface, expert_demos to generate dataset points
"""
parser = argparse.ArgumentParser(usage=usage)
parser.add_argument("h5file", type=str)
parser.add_argument("--cloud_proc_func", default="extract_red")
parser.add_argument("--cloud_proc_mod", default="rapprentice.cloud_proc_funcs")
    
parser.add_argument("--execution", type=int, default=0)
parser.add_argument("--animation", type=int, default=0)
parser.add_argument("--simulation", type=int, default=0)
parser.add_argument("--parallel", type=int, default=1)

parser.add_argument("--prompt", action="store_true")
parser.add_argument("--select_manual", action="store_true")

parser.add_argument("--fake_data_segment",type=str)
parser.add_argument("--fake_data_transform", type=float, nargs=6, metavar=("tx","ty","tz","rx","ry","rz"),
    default=[0,0,0,0,0,0], help="translation=(tx,ty,tz), axis-angle rotation=(rx,ry,rz)")
parser.add_argument("--sim_init_perturb_radius", type=float, default=None)
parser.add_argument("--sim_init_perturb_num_points", type=int, default=7)
parser.add_argument("--sim_desired_knot_name", type=str, default=None)
parser.add_argument("--max_steps_before_failure", type=int, default=-1)
parser.add_argument("--random_seed", type=int, default=None)
parser.add_argument("--no_failure_examples", type=int, default=0)
parser.add_argument("--only_first_n_examples", type=int, default=-1)
parser.add_argument("--only_examples_from_list", type=str)
parser.add_argument("--lookahead", type=int, default=1)
parser.add_argument("--interactive",action="store_true")
parser.add_argument("--log", type=str, default="", help="")
parser.add_argument("--results_outfile", type=str, default=None)

args = parser.parse_args()

if args.fake_data_segment is None: assert args.execution==1
if args.simulation: assert args.execution == 0 and args.fake_data_segment is not None

###################


"""
Workflow:
1. Fake data + animation only
    --fake_data_segment=xxx --execution=0
2. Fake data + Gazebo. Set Gazebo to initial state of fake data segment so we'll execute the same thing.
    --fake_data_segment=xxx --execution=1
    This is just so we know the robot won't do something stupid that we didn't catch with openrave only mode.
3. Real data + Gazebo
    --execution=1 
    The problem is that the gazebo robot is in a different state from the real robot, in particular, the head tilt angle. TODO: write a script that       sets gazebo head to real robot head
4. Real data + Real execution.
    --execution=1

The question is, do you update the robot's head transform.
If you're using fake data, don't update it.

"""

from rapprentice import registration, colorize, berkeley_pr2, \
     animate_traj, ros2rave, plotting_openrave, task_execution, \
     planning, tps, func_utils, resampling, ropesim, rope_initialization, clouds
from rapprentice import math_utils as mu
from rapprentice.yes_or_no import yes_or_no
import pdb, time

try:
    from rapprentice import pr2_trajectories, PR2
    import rospy
except ImportError:
    print "Couldn't import ros stuff"
if args.parallel:
    from joblib import Parallel, delayed

import cloudprocpy, trajoptpy, openravepy
import os, numpy as np, h5py
from numpy import asarray
import atexit
import importlib
from itertools import combinations
import IPython as ipy
import random

cloud_proc_mod = importlib.import_module(args.cloud_proc_mod)
cloud_proc_func = getattr(cloud_proc_mod, args.cloud_proc_func)

if args.random_seed is not None: np.random.seed(args.random_seed)
    
    
def redprint(msg):
    print colorize.colorize(msg, "red", bold=True)
    
def split_trajectory_by_gripper(seg_info):
    rgrip = asarray(seg_info["r_gripper_joint"])
    lgrip = asarray(seg_info["l_gripper_joint"])

    thresh = .04 # open/close threshold

    n_steps = len(lgrip)


    # indices BEFORE transition occurs
    l_openings = np.flatnonzero((lgrip[1:] >= thresh) & (lgrip[:-1] < thresh))
    r_openings = np.flatnonzero((rgrip[1:] >= thresh) & (rgrip[:-1] < thresh))
    l_closings = np.flatnonzero((lgrip[1:] < thresh) & (lgrip[:-1] >= thresh))
    r_closings = np.flatnonzero((rgrip[1:] < thresh) & (rgrip[:-1] >= thresh))

    before_transitions = np.r_[l_openings, r_openings, l_closings, r_closings]
    after_transitions = before_transitions+1
    seg_starts = np.unique(np.r_[0, after_transitions])
    seg_ends = np.unique(np.r_[before_transitions, n_steps-1])

    return seg_starts, seg_ends

def binarize_gripper(angle):
    thresh = .04
    return angle > thresh
    
def set_gripper_maybesim(lr, is_open, prev_is_open):
    mult = 1 if args.execution else 5
    open_angle = .08 * mult
    closed_angle = (0 if not args.simulation else .02) * mult

    target_val = open_angle if is_open else closed_angle

    if args.execution:
        gripper = {"l":Globals.pr2.lgrip, "r":Globals.pr2.rgrip}[lr]
        gripper.set_angle(target_val)
        Globals.pr2.join_all()

    elif not args.simulation:
        Globals.robot.SetDOFValues([target_val], [Globals.robot.GetJoint("%s_gripper_l_finger_joint"%lr).GetDOFIndex()])

    elif args.simulation:
        # release constraints if necessary
        if is_open and not prev_is_open:
            Globals.sim.release_rope(lr)
            print "DONE RELEASING"

        # execute gripper open/close trajectory
        joint_ind = Globals.robot.GetJoint("%s_gripper_l_finger_joint"%lr).GetDOFIndex()
        start_val = Globals.robot.GetDOFValues([joint_ind])[0]
        joint_traj = np.linspace(start_val, target_val, np.ceil(abs(target_val - start_val) / .02))
        for val in joint_traj:
            Globals.robot.SetDOFValues([val], [joint_ind])
            Globals.sim.step()
            if args.animation:
#                Globals.viewer.Step()
                if args.interactive: Globals.viewer.Idle()
        # add constraints if necessary
        Globals.viewer.Step()
        if not is_open and prev_is_open:
            if not Globals.sim.grab_rope(lr):
                return False

    return True

def unwrap_arm_traj_in_place(traj):
    assert traj.shape[1] == 7
    for i in [2,4,6]:
        traj[:,i] = np.unwrap(traj[:,i])
    return traj

def unwrap_in_place(t):
    # TODO: do something smarter than just checking shape[1]
    if t.shape[1] == 7:
        unwrap_arm_traj_in_place(t)
    elif t.shape[1] == 14:
        unwrap_arm_traj_in_place(t[:,:7])
        unwrap_arm_traj_in_place(t[:,7:])
    else:
        raise NotImplementedError

def exec_traj_noanimate(bodypart2traj):
    (args_simulation, args_animation) = (args.simulation, args.animation)
    args.animation = False
    result = exec_traj_maybesim(bodypart2traj)
    (args.simulation, args.animation) = (args_simulation, args_animation)
    return result
    

def exec_traj_maybesim(bodypart2traj):
    def sim_callback(i):
        Globals.sim.step()

    animate_speed = 10 if args.animation else 0

    if args.animation or args.simulation:
        dof_inds = []
        trajs = []
        for (part_name, traj) in bodypart2traj.items():
            manip_name = {"larm":"leftarm","rarm":"rightarm"}[part_name]
            dof_inds.extend(Globals.robot.GetManipulator(manip_name).GetArmIndices())            
            trajs.append(traj)
        full_traj = np.concatenate(trajs, axis=1)
        Globals.robot.SetActiveDOFs(dof_inds)

        if args.simulation:
            # make the trajectory slow enough for the simulation
            full_traj = ropesim.retime_traj(Globals.robot, dof_inds, full_traj)

            # in simulation mode, we must make sure to gradually move to the new starting position
            curr_vals = Globals.robot.GetActiveDOFValues()
            transition_traj = np.r_[[curr_vals], [full_traj[0]]]
            unwrap_in_place(transition_traj)
            transition_traj = ropesim.retime_traj(Globals.robot, dof_inds, transition_traj, max_cart_vel=.05)
            animate_traj.animate_traj(transition_traj, Globals.robot, restore=False, pause=args.interactive,
                callback=sim_callback if args.simulation else None, step_viewer=animate_speed)
            full_traj[0] = transition_traj[-1]
            unwrap_in_place(full_traj)

        animate_traj.animate_traj(full_traj, Globals.robot, restore=False, pause=args.interactive,
            callback=sim_callback if args.simulation else None, step_viewer=animate_speed)
        Globals.viewer.Step()
        return True

    if args.execution:
        if not args.prompt or yes_or_no("execute?"):
            pr2_trajectories.follow_body_traj(Globals.pr2, bodypart2traj)
        else:
            return False

    return True

def load_random_start_segment(demofile):
    start_keys = [k for k in demofile.keys() if k.startswith('demo') and k.endswith('00')]
    seg_name = random.choice(start_keys)
    return demofile[seg_name]['cloud_xyz']

def sample_rope_state(demofile, human_check=True, perturb_points=5, min_rad=0, max_rad=.15):
    success = False
    while not success:
        # TODO: pick a random rope initialization
        new_xyz= load_random_start_segment(demofile)
        perturb_radius = random.uniform(min_rad, max_rad)
        rope_nodes = rope_initialization.find_path_through_point_cloud( new_xyz,
                                                                        perturb_peak_dist=perturb_radius,
                                                                        num_perturb_points=perturb_points)
        replace_rope(rope_nodes)
        Globals.sim.settle()
        Globals.viewer.Step()
        if human_check:
            resp = raw_input("Use this simulation?[Y/n]")
            success = resp not in ('N', 'n')
        else:
            success = True


def find_closest_manual(demofile, new_xyz, outfile):
    "for now, just prompt the user"
    seg_names = demofile.keys()
    print "choose from the following options (type an integer)"
    ignore_inds = get_ignored_inds(demofile)
    keys = remove_inds(demofile.keys(), ignore_inds)
    ds_clouds = dict(zip(keys, remove_inds(get_downsampled_clouds(demofile), ignore_inds)))
    ds_new = clouds.downsample(new_xyz,DS_SIZE)
    if args.parallel:
        ds_items = sorted(ds_clouds.items())
        costs = Parallel(n_jobs=-1,verbose=100)(delayed(registration_cost)(ds_cloud, ds_new) for (s_name, ds_cloud) in ds_items)
        names_costs = [(ds_items[i][0], costs[i]) for i in range(len(ds_items))]
        costs = dict(names_costs)
    else:
        costs = {}
        for i, (seg_name, ds_cloud) in enumerate(ds_clouds.items()):
            costs[seg_name] = registration_cost(ds_cloud, ds_new)
            print "completed %i/%i"%(i+1, len(ds_clouds))
    best_keys = sorted(costs, key=costs.get)
    rope_state = Globals.sim.rope.GetControlPoints()
    for seg_name in best_keys:
        # Set animate=False to speed up collection
        # pass back result = None --> ignore this one
        (success, result) = simulate_demo(new_xyz, demofile, seg_name, reset_rope=rope_state, animate=True, pause=True)        
        if success:
            break
    if result != None: # TODO: save pt cld and action
        save_example_action(outfile, new_xyz, seg_name)
    return (seg_name, False)

def save_example_action(filename, xyz, action):
    with h5py.File(filename, 'a') as f:
        if f.keys():
            "new key just adds one to the largest index"
            new_k = str(sorted([int(k) for k in f.keys()])[-1] + 1) 
        else:
            "new key is just 0"
            new_k = '0'
        f.create_group(new_k)
        f[new_k]['cloud_xyz'] = xyz
        f[new_k]['action'] = str(action)
        f.close()

def registration_cost(xyz0, xyz1):
    scaled_xyz0, _ = registration.unit_boxify(xyz0)
    scaled_xyz1, _ = registration.unit_boxify(xyz1)
    f,g = registration.tps_rpm_bij(scaled_xyz0, scaled_xyz1, rot_reg=1e-3, n_iter=10)
    cost = registration.tps_reg_cost(f) + registration.tps_reg_cost(g)
    return cost

DS_SIZE = .025

@func_utils.once
def get_ignored_inds(demofile):
    ignore = []
    for i, k in enumerate(demofile.keys()):
        if args.no_failure_examples and "failure" in k:
            ignore.append(i)
        elif args.only_first_n_examples != -1 and k.startswith("demo"):
            curr_num = int(k[len("demo"):].split("-")[0])
            if curr_num > args.only_first_n_examples:
                ignore.append(i)
        elif args.only_examples_from_list and k.startswith("demo"):
            allowed_examples = set(map(int, args.only_examples_from_list.split(',')))
            curr_num = int(k[len("demo"):].split("-")[0])
            if curr_num not in allowed_examples:
                ignore.append(i)
    print 'Ignoring examples:', [k for (i, k) in enumerate(demofile.keys()) if i in ignore]
    return np.asarray(ignore)

@func_utils.once
def get_downsampled_clouds(demofile):
    return [clouds.downsample(seg["cloud_xyz"], DS_SIZE) for seg in demofile.values()]

def remove_inds(a, inds):
    return [x for (i, x) in enumerate(a) if i not in inds]

def find_closest_auto(demofile, new_xyz):
    cluster_info = kmeans.load_clusters('clusters.pkl')
    reverse_ind = cluster_info['reverse_ind']
    ignore_inds = get_ignored_inds(demofile)
    keys = remove_inds(demofile.keys(), ignore_inds)
    ds_clouds = dict(zip(keys, remove_inds(get_downsampled_clouds(demofile), ignore_inds)))
    ds_new = clouds.downsample(new_xyz,DS_SIZE)
    if args.parallel:
        ds_items = sorted(ds_clouds.items())
        costs = Parallel(n_jobs=-1,verbose=100)(delayed(registration_cost)(ds_cloud, ds_new) for (s_name, ds_cloud) in ds_items)
        names_costs = [(ds_items[i][0], costs[i]) for i in range(len(ds_items))]
        costs = dict(names_costs)
    else:
        costs = {}
        for i, (seg_name, ds_cloud) in enumerate(ds_clouds.items()):
            # success, sim_xyz = simulate_demo(new_xyz, demofile, seg_name, animate=False)
            # ds_sim = clouds.downsample(sim_xyz, DS_SIZE)
            # seg_num = int(seg_name[-1])##coder beware --- super brittle
            # next_seg = seg_name[:-1] + str(seg_num+1)
            # if next_seg in ds_clouds:
            #    costs.append(registration_cost(ds_clouds[next_seg], ds_sim));
            # else:
            costs[seg_name] = registration_cost(ds_cloud, ds_new)
            print "completed %i/%i"%(i+1, len(ds_clouds))
    best_keys = sorted(costs, key=costs.get)[:10]
    if best_keys[0].startswith('endstate'):
        return best_keys[0]
    best_cluster = reverse_ind[best_keys[0]]
    rv_ind_safe = lambda k: reverse_ind[k] if k in reverse_ind else -1
    consensus_cost = sum(rv_ind_safe(key) != best_cluster for key in best_keys)
    if args.lookahead and consensus_cost:
        N = 10
        CONSENSUS_N = 15
        rope_state = Globals.sim.rope.GetControlPoints()
        top_n_sorted_costs = [(costs[key], key) for key in sorted(costs, key=costs.get)[:N]]
        #top_n_sorted_costs = sorted(zip(costs, keys))[:N]
        final_costs = []
        for i, (cost, seg_name) in enumerate(top_n_sorted_costs):
            ds_cloud = ds_clouds[seg_name]
            success, sim_xyz = simulate_demo(new_xyz, demofile, seg_name, reset_rope=rope_state, animate=False)
            if not success:
                final_costs.append((9999, 99999, 9999))#append infinity
                continue
            ds_sim = clouds.downsample(sim_xyz, DS_SIZE)
            print "Lookahead {}/{}...".format(i+1, N)
            costs_2 = {}
            if args.parallel:
                ds_items = sorted(ds_clouds.items())
                costs2 = Parallel(n_jobs=-1,verbose=100)(delayed(registration_cost)(ds_cloud, ds_new) for (s_name, ds_cloud) in ds_items)
                names_costs = [(ds_items[i][0], costs2[i]) for i in range(len(ds_items))]
                costs_2 = dict(names_costs)
            else:
                for seg_name_2, ds_cloud_2 in ds_clouds.items():
                    costs_2[seg_name_2] = registration_cost(ds_cloud_2, ds_sim)
            best_keys = sorted(costs_2, key=costs_2.get)[:CONSENSUS_N]
            best_cluster = rv_ind_safe(best_keys[0])
            consensus_cost = sum(rv_ind_safe(key) != best_cluster for key in best_keys)
            final_costs.append((consensus_cost, registration_cost(ds_cloud, ds_new),
                rv_ind_safe(seg_name)))
        ibest = tuple_argmin(final_costs)
        pprint.pprint(sorted(final_costs))        
        if ibest >= len(top_n_sorted_costs):
            print "ibest out of range"
            ipy.embed()       
        return top_n_sorted_costs[ibest][1]
    else:
        #print "costs\n", costs
        #pprint.pprint(sorted(costs.values()))
        return min(costs, key=costs.get)

def tuple_argmin(l):
    best = None
    min_val = l[0]
    for (i, t) in enumerate(l):
        if t <=min_val:
            min_val = t
            best = i
    return best

def simulate_demo(new_xyz, demofile, seg_name, reset_rope=None, animate=False, pause=False):
    Globals.robot.SetDOFValues(PR2_L_POSTURES["side"], Globals.robot.GetManipulator("leftarm").GetArmIndices())
    Globals.robot.SetDOFValues(mirror_arm_joints(PR2_L_POSTURES["side"]), Globals.robot.GetManipulator("rightarm").GetArmIndices())
    seg_info = demofile[seg_name]
    
    redprint("Generating end-effector trajectory")    
    
    handles = []
    old_xyz = np.squeeze(seg_info["cloud_xyz"])
    handles.append(Globals.env.plot3(old_xyz,5, (1,0,0)))
    handles.append(Globals.env.plot3(new_xyz,5, (0,0,1)))
    
    old_xyz = clouds.downsample(old_xyz, DS_SIZE)
    new_xyz = clouds.downsample(new_xyz, DS_SIZE)
    
    scaled_old_xyz, src_params = registration.unit_boxify(old_xyz)
    scaled_new_xyz, targ_params = registration.unit_boxify(new_xyz)        
    f,_ = registration.tps_rpm_bij(scaled_old_xyz, scaled_new_xyz, plot_cb = tpsrpm_plot_cb,
                                   plotting=5 if args.animation else 0,rot_reg=np.r_[1e-4,1e-4,1e-1], n_iter=50, reg_init=10, reg_final=.1)
    f = registration.unscale_tps(f, src_params, targ_params)

    handles.extend(plotting_openrave.draw_grid(Globals.env, f.transform_points, old_xyz.min(axis=0)-np.r_[0,0,.1], old_xyz.max(axis=0)+np.r_[0,0,.1], xres = .1, yres = .1, zres = .04))        

    link2eetraj = {}
    for lr in 'lr':
        link_name = "%s_gripper_tool_frame"%lr
        old_ee_traj = asarray(seg_info[link_name]["hmat"])
        new_ee_traj = f.transform_hmats(old_ee_traj)
        link2eetraj[link_name] = new_ee_traj
        
        handles.append(Globals.env.drawlinestrip(old_ee_traj[:,:3,3], 2, (1,0,0,1)))
        handles.append(Globals.env.drawlinestrip(new_ee_traj[:,:3,3], 2, (0,1,0,1)))

    miniseg_starts, miniseg_ends = split_trajectory_by_gripper(seg_info)    
    success = True
    print colorize.colorize("mini segments:", "red"), miniseg_starts, miniseg_ends
    for (i_miniseg, (i_start, i_end)) in enumerate(zip(miniseg_starts, miniseg_ends)):            

        ################################    
        redprint("Generating joint trajectory for segment %s, part %i"%(seg_name, i_miniseg))

        # figure out how we're gonna resample stuff
        lr2oldtraj = {}
        for lr in 'lr':
            manip_name = {"l":"leftarm", "r":"rightarm"}[lr]                 
            old_joint_traj = asarray(seg_info[manip_name][i_start:i_end+1])
            #print (old_joint_traj[1:] - old_joint_traj[:-1]).ptp(axis=0), i_start, i_end
            if arm_moved(old_joint_traj):       
                lr2oldtraj[lr] = old_joint_traj   
        if len(lr2oldtraj) > 0:
            old_total_traj = np.concatenate(lr2oldtraj.values(), 1)
            JOINT_LENGTH_PER_STEP = .1
            _, timesteps_rs = unif_resample(old_total_traj, JOINT_LENGTH_PER_STEP)
        ####

        ### Generate fullbody traj
        bodypart2traj = {}            
        for (lr,old_joint_traj) in lr2oldtraj.items():
            
            manip_name = {"l":"leftarm", "r":"rightarm"}[lr]
            
            old_joint_traj_rs = mu.interp2d(timesteps_rs, np.arange(len(old_joint_traj)), old_joint_traj)
            
            ee_link_name = "%s_gripper_tool_frame"%lr
            new_ee_traj = link2eetraj[ee_link_name][i_start:i_end+1]          
            new_ee_traj_rs = resampling.interp_hmats(timesteps_rs, np.arange(len(new_ee_traj)), new_ee_traj)
            if args.execution: Globals.pr2.update_rave()
            new_joint_traj = planning.plan_follow_traj(Globals.robot, manip_name,
                                                       Globals.robot.GetLink(ee_link_name), new_ee_traj_rs,old_joint_traj_rs)
            part_name = {"l":"larm", "r":"rarm"}[lr]
            bodypart2traj[part_name] = new_joint_traj
            ################################    
            redprint("Executing joint trajectory for segment %s, part %i using arms '%s'"%(seg_name, i_miniseg, bodypart2traj.keys()))

        for lr in 'lr':
            gripper_open = binarize_gripper(seg_info["%s_gripper_joint"%lr][i_start])
            prev_gripper_open = binarize_gripper(seg_info["%s_gripper_joint"%lr][i_start-1]) if i_start != 0 else False
            if not set_gripper_maybesim(lr, gripper_open, prev_gripper_open):
                redprint("Grab %s failed" % lr)
                success = False

        if not success: break

        if len(bodypart2traj) > 0:
            if animate:
                success &= exec_traj_maybesim(bodypart2traj)
            else:
                success &= exec_traj_noanimate(bodypart2traj)

        if not success: break


    Globals.sim.settle(animate=args.animation)
    result_xyz = Globals.sim.observe_cloud()
    Globals.robot.SetDOFValues(PR2_L_POSTURES["side"], Globals.robot.GetManipulator("leftarm").GetArmIndices())
    Globals.robot.SetDOFValues(mirror_arm_joints(PR2_L_POSTURES["side"]), Globals.robot.GetManipulator("rightarm").GetArmIndices())

    if pause:
        print "d accepts and resamples rope"
        print "i ignores and resamples rope"
        response = raw_input("Use this demonstration?[y/N/d/i]")
        if response in ('I', 'i'):
            Globals.resample_rope = True
            return (True, None)
        if response in ('D', 'd'):
            Globals.resample_rope = True
            return (True, result_xyz)
        success = response in ('Y', 'y')
        if success:
            return (success, result_xyz)

    # print "Grab Screen Shot Now"
    # Globals.viewer.Idle()
    Globals.sim.release_rope('l')
    Globals.sim.release_rope('r')
    if reset_rope!=None:        
        replace_rope(reset_rope)
        Globals.sim.settle()
        if np.linalg.norm(Globals.sim.rope.GetControlPoints() - reset_rope) > .1:
            print "uh oh, new rope is different..."
            # pdb.set_trace()
    
    return success, result_xyz

def replace_rope(new_rope):
    import bulletsimpy
    old_rope_nodes = Globals.sim.rope.GetControlPoints()
    Globals.viewer.RemoveKinBody(Globals.env.GetKinBody('rope'))
    Globals.env.Remove(Globals.env.GetKinBody('rope'))
    Globals.sim.bt_env.Remove(Globals.sim.bt_env.GetObjectByName('rope'))
    Globals.sim.rope = bulletsimpy.CapsuleRope(Globals.sim.bt_env, 'rope', new_rope,
                                               Globals.sim.rope_params)
    return old_rope_nodes
    


def arm_moved(joint_traj):    
    if len(joint_traj) < 2: return False
    return ((joint_traj[1:] - joint_traj[:-1]).ptp(axis=0) > .01).any()
        
def tpsrpm_plot_cb(x_nd, y_md, targ_Nd, corr_nm, wt_n, f):
    ypred_nd = f.transform_points(x_nd)
    handles = []
    handles.append(Globals.env.plot3(ypred_nd, 3, (0,1,0,1)))
    handles.extend(plotting_openrave.draw_grid(Globals.env, f.transform_points, x_nd.min(axis=0), x_nd.max(axis=0), xres = .1, yres = .1, zres = .04))
    if Globals.viewer:
        Globals.viewer.Step()

def load_fake_data_segment(demofile, set_robot_state=True):
    fake_seg = demofile[args.fake_data_segment]
    new_xyz = np.squeeze(fake_seg["cloud_xyz"])
    hmat = openravepy.matrixFromAxisAngle(args.fake_data_transform[3:6])
    hmat[:3,3] = args.fake_data_transform[0:3]
    new_xyz = new_xyz.dot(hmat[:3,:3].T) + hmat[:3,3][None,:]
    r2r = ros2rave.RosToRave(Globals.robot, asarray(fake_seg["joint_states"]["name"]))
    if set_robot_state:
        r2r.set_values(Globals.robot, asarray(fake_seg["joint_states"]["position"][0]))
    return new_xyz, r2r

def unif_resample(traj, max_diff, wt = None):        
    """
    Resample a trajectory so steps have same length in joint space    
    """
    import scipy.interpolate as si
    tol = .005
    if wt is not None: 
        wt = np.atleast_2d(wt)
        traj = traj*wt
        
        
    dl = mu.norms(traj[1:] - traj[:-1],1)
    l = np.cumsum(np.r_[0,dl])
    goodinds = np.r_[True, dl > 1e-8]
    deg = min(3, sum(goodinds) - 1)
    if deg < 1: return traj, np.arange(len(traj))
    
    nsteps = max(int(np.ceil(float(l[-1])/max_diff)), 2)
    newl = np.linspace(0,l[-1],nsteps)

    ncols = traj.shape[1]
    colstep = 10
    traj_rs = np.empty((nsteps,ncols)) 
    for istart in xrange(0, traj.shape[1], colstep):
        (tck,_) = si.splprep(traj[goodinds, istart:istart+colstep].T,k=deg,s = tol**2*len(traj),u=l[goodinds])
        traj_rs[:,istart:istart+colstep] = np.array(si.splev(newl,tck)).T
    if wt is not None: traj_rs = traj_rs/wt

    newt = np.interp(newl, l, np.arange(len(traj)))

    return traj_rs, newt

def make_table_xml(translation, extents):
    xml = """
<Environment>
  <KinBody name="table">
    <Body type="static" name="table_link">
      <Geom type="box">
        <Translation>%f %f %f</Translation>
        <extents>%f %f %f</extents>
        <diffuseColor>.96 .87 .70</diffuseColor>
      </Geom>
    </Body>
  </KinBody>
</Environment>
""" % (translation[0], translation[1], translation[2], extents[0], extents[1], extents[2])
    return xml

PR2_L_POSTURES = dict(
    untucked = [0.4,  1.0,   0.0,  -2.05,  0.0,  -0.1,  0.0],
    tucked = [0.06, 1.25, 1.79, -1.68, -1.73, -0.10, -0.09],
    up = [ 0.33, -0.35,  2.59, -0.15,  0.59, -1.41, -0.27],
    side = [  1.832,  -0.332,   1.011,  -1.437,   1.1  ,  -2.106,  3.074]
)
def mirror_arm_joints(x):
    "mirror image of joints (r->l or l->r)"
    return np.r_[-x[0],x[1],-x[2],x[3],-x[4],x[5],-x[6]]

###################

class Globals:
    robot = None
    env = None
    pr2 = None
    sim = None
    log = None
    viewer = None
    resample_rope = None

def main():

    demofile = h5py.File(args.h5file, 'r')
    
    trajoptpy.SetInteractive(args.interactive)

    if not args.log:
        from datetime import datetime
        args.log = "log_%s.pkl" % datetime.now().isoformat()
    redprint("Writing log to file %s" % args.log)
    Globals.exec_log = task_execution.ExecutionLog(args.log)
    atexit.register(Globals.exec_log.close)

    Globals.exec_log(0, "main.args", args)

    if args.execution:
        rospy.init_node("exec_task",disable_signals=True)
        Globals.pr2 = PR2.PR2()
        Globals.env = Globals.pr2.env
        Globals.robot = Globals.pr2.robot
    else:
        Globals.env = openravepy.Environment()
        Globals.env.StopSimulation()
        Globals.env.Load("robots/pr2-beta-static.zae")
        Globals.robot = Globals.env.GetRobots()[0]
        if args.simulation:
            init_rope_xyz, _ = load_fake_data_segment(demofile, set_robot_state=False)
            table_height = init_rope_xyz[:,2].mean() - .02
            table_xml = make_table_xml(translation=[1, 0, table_height], extents=[.85, .55, .01])
            Globals.env.LoadData(table_xml)
            Globals.sim = ropesim.Simulation(Globals.env, Globals.robot)

    if not args.fake_data_segment:
        grabber = cloudprocpy.CloudGrabber()
        grabber.startRGBD()

    if args.animation:
        Globals.viewer = trajoptpy.GetViewer(Globals.env)

    #####################
    curr_step = 0


    while True:
        # if args.max_steps_before_failure != -1 and curr_step > args.max_steps_before_failure:
        #     redprint("Number of steps %d exceeded maximum %d" % (curr_step, args.max_steps_before_failure))
        #     Globals.exec_log(curr_step, "result", False, description="step maximum reached")
        #     break        

        curr_step += 1
    
        if Globals.resample_rope:            
            sample_rope_state(demofile, human_check=False)
            Globals.resample_rope = False

        redprint("Acquire point cloud")

        if args.simulation:
            Globals.robot.SetDOFValues(PR2_L_POSTURES["side"], Globals.robot.GetManipulator("leftarm").GetArmIndices())
            Globals.robot.SetDOFValues(mirror_arm_joints(PR2_L_POSTURES["side"]), Globals.robot.GetManipulator("rightarm").GetArmIndices())

            if curr_step == 1:
                # create rope
                new_xyz, r2r = load_fake_data_segment(demofile)
                Globals.exec_log(curr_step, "acquire_cloud.orig_cloud", new_xyz)
                rope_nodes = rope_initialization.find_path_through_point_cloud(
                    new_xyz,
                    perturb_peak_dist=args.sim_init_perturb_radius,
                    num_perturb_points=args.sim_init_perturb_num_points)
                Globals.exec_log(curr_step, "acquire_cloud.init_sim_rope_nodes", rope_nodes)
                Globals.sim.create(rope_nodes)
                print "move viewer to viewpoint that isn't stupid"
                print "then hit 'p' to continue"
                Globals.viewer.Idle()
                sample_rope_state(demofile)
            #elif curr_step == 2:
            #    step_2_rope = Globals.sim.rope.GetNodes()
            #    replace_rope(step_2_rope)
            #elif curr_step == 3:
            #    replace_rope(step_2_rope)

            new_xyz = Globals.sim.observe_cloud()
            


        elif args.fake_data_segment:
            new_xyz, r2r = load_fake_data_segment(demofile)


        else:
            Globals.pr2.rarm.goto_posture('side')
            Globals.pr2.larm.goto_posture('side')            
            Globals.pr2.join_all()
            
            Globals.pr2.update_rave()
            
            rgb, depth = grabber.getRGBD()
            T_w_k = berkeley_pr2.get_kinect_transform(Globals.robot)
            new_xyz = cloud_proc_func(rgb, depth, T_w_k)
        Globals.exec_log(curr_step, "acquire_cloud.xyz", new_xyz)
        handles = []
        ################################    
        redprint("Finding closest demonstration")
        if args.select_manual:
            (seg_name, execute) = find_closest_manual(demofile, new_xyz, args.results_outfile)
        else:
            seg_name = find_closest_auto(demofile, new_xyz)

        if not execute: continue

        seg_info = demofile[seg_name]
        # redprint("using demo %s, description: %s"%(seg_name, seg_info["description"]))
        Globals.exec_log(curr_step, "find_closest_demo.seg_name", seg_name)

        if "endstates" in seg_name:
            redprint("Recognized end state %s. Done!" % seg_name)
            Globals.exec_log(curr_step, "result", True, description="end state %s" % seg_name)
            break
    
        ################################

        redprint("Generating end-effector trajectory")    


        old_xyz = np.squeeze(seg_info["cloud_xyz"])
        handles.append(Globals.env.plot3(old_xyz,5, (1,0,0)))
        handles.append(Globals.env.plot3(new_xyz,5, (0,0,1)))
        
        old_xyz = clouds.downsample(old_xyz, DS_SIZE)
        new_xyz = clouds.downsample(new_xyz, DS_SIZE)

        scaled_old_xyz, src_params = registration.unit_boxify(old_xyz)
        scaled_new_xyz, targ_params = registration.unit_boxify(new_xyz)        
        f,_ = registration.tps_rpm_bij(scaled_old_xyz, scaled_new_xyz, plot_cb = tpsrpm_plot_cb,
                                       plotting=5 if args.animation else 0,rot_reg=np.r_[1e-4,1e-4,1e-1], n_iter=50, reg_init=10, reg_final=.1)
        f = registration.unscale_tps(f, src_params, targ_params)
        Globals.exec_log(curr_step, "gen_traj.f", f)
        
        handles.extend(plotting_openrave.draw_grid(Globals.env, f.transform_points, old_xyz.min(axis=0)-np.r_[0,0,.1], old_xyz.max(axis=0)+np.r_[0,0,.1], xres = .1, yres = .1, zres = .04))        

        link2eetraj = {}
        for lr in 'lr':
            link_name = "%s_gripper_tool_frame"%lr
            old_ee_traj = asarray(seg_info[link_name]["hmat"])
            new_ee_traj = f.transform_hmats(old_ee_traj)
            link2eetraj[link_name] = new_ee_traj
            
            handles.append(Globals.env.drawlinestrip(old_ee_traj[:,:3,3], 2, (1,0,0,1)))
            handles.append(Globals.env.drawlinestrip(new_ee_traj[:,:3,3], 2, (0,1,0,1)))
        Globals.exec_log(curr_step, "gen_traj.link2eetraj", link2eetraj)

        miniseg_starts, miniseg_ends = split_trajectory_by_gripper(seg_info)    
        success = True
        print colorize.colorize("mini segments:", "red"), miniseg_starts, miniseg_ends
        for (i_miniseg, (i_start, i_end)) in enumerate(zip(miniseg_starts, miniseg_ends)):
            
            if args.execution=="real": Globals.pr2.update_rave()


            ################################    
            redprint("Generating joint trajectory for segment %s, part %i"%(seg_name, i_miniseg))

            # figure out how we're gonna resample stuff
            lr2oldtraj = {}
            for lr in 'lr':
                manip_name = {"l":"leftarm", "r":"rightarm"}[lr]                 
                old_joint_traj = asarray(seg_info[manip_name][i_start:i_end+1])
                #print (old_joint_traj[1:] - old_joint_traj[:-1]).ptp(axis=0), i_start, i_end
                if arm_moved(old_joint_traj):       
                    lr2oldtraj[lr] = old_joint_traj   
            if len(lr2oldtraj) > 0:
                old_total_traj = np.concatenate(lr2oldtraj.values(), 1)
                JOINT_LENGTH_PER_STEP = .1
                _, timesteps_rs = unif_resample(old_total_traj, JOINT_LENGTH_PER_STEP)
            ####

            ### Generate fullbody traj
            bodypart2traj = {}            
            for (lr,old_joint_traj) in lr2oldtraj.items():

                manip_name = {"l":"leftarm", "r":"rightarm"}[lr]
                 
                old_joint_traj_rs = mu.interp2d(timesteps_rs, np.arange(len(old_joint_traj)), old_joint_traj)

                ee_link_name = "%s_gripper_tool_frame"%lr
                new_ee_traj = link2eetraj[ee_link_name][i_start:i_end+1]          
                new_ee_traj_rs = resampling.interp_hmats(timesteps_rs, np.arange(len(new_ee_traj)), new_ee_traj)
                if args.execution: Globals.pr2.update_rave()
                new_joint_traj = planning.plan_follow_traj(Globals.robot, manip_name,
                 Globals.robot.GetLink(ee_link_name), new_ee_traj_rs,old_joint_traj_rs)
                part_name = {"l":"larm", "r":"rarm"}[lr]
                bodypart2traj[part_name] = new_joint_traj

        

            ################################    
            redprint("Executing joint trajectory for segment %s, part %i using arms '%s'"%(seg_name, i_miniseg, bodypart2traj.keys()))

            for lr in 'lr':
                gripper_open = binarize_gripper(seg_info["%s_gripper_joint"%lr][i_start])
                prev_gripper_open = binarize_gripper(seg_info["%s_gripper_joint"%lr][i_start-1]) if i_start != 0 else False
                if not set_gripper_maybesim(lr, gripper_open, prev_gripper_open):
                    redprint("Grab %s failed" % lr)
                    success = False
            if args.simulation:
                Globals.exec_log(curr_step, "execute_traj.miniseg_%d.sim_rope_nodes_after_grab"%i_miniseg, Globals.sim.rope.GetNodes())

            if not success: break

            if len(bodypart2traj) > 0:
                success &= exec_traj_maybesim(bodypart2traj)
            if args.simulation:
                Globals.exec_log(curr_step, "execute_traj.miniseg_%d.sim_rope_nodes_after_traj"%i_miniseg, Globals.sim.rope.GetNodes())

            if not success: break

        if args.simulation:
            Globals.sim.settle(animate=args.animation)
            Globals.exec_log(curr_step, "execute_traj.sim_rope_nodes_after_full_traj", Globals.sim.rope.GetNodes())

            if args.sim_desired_knot_name is not None:
                from rapprentice import knot_identification
                knot_name = knot_identification.identify_knot(Globals.sim.rope.GetControlPoints())
                if knot_name is not None:
                    if knot_name == args.sim_desired_knot_name or args.sim_desired_knot_name == "any":
                        redprint("Identified knot: %s. Success!" % knot_name)
                        Globals.exec_log(curr_step, "result", True, description="identified knot %s" % knot_name)
                        break
                    else:
                        redprint("Identified knot: %s, but expected %s. Continuing." % (knot_name, args.sim_desired_knot_name))
                else:
                    redprint("Not a knot. Continuing.")

        redprint("Segment %s result: %s"%(seg_name, success))

        if args.fake_data_segment and not args.simulation: break

if __name__ == "__main__":
    main()
