#!/usr/bin/env python

import pprint
import argparse

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

import cloudprocpy, trajoptpy, openravepy
import util
from rope_qlearn import *
import os, numpy as np, h5py
from numpy import asarray
import atexit
import importlib
from itertools import combinations
import IPython as ipy
import random
import sys

from joblib import Parallel, delayed

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
    mult = 5
    open_angle = .08 * mult
    closed_angle = .02 * mult

    target_val = open_angle if is_open else closed_angle

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
#         if args.animation:
#                Globals.viewer.Step()
#             if args.interactive: Globals.viewer.Idle()
    # add constraints if necessary
    if Globals.viewer:
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

def sim_traj_maybesim(bodypart2traj, animate=False, interactive=False):
    def sim_callback(i):
        Globals.sim.step()

    animate_speed = 10 if animate else 0

    dof_inds = []
    trajs = []
    for (part_name, traj) in bodypart2traj.items():
        manip_name = {"larm":"leftarm","rarm":"rightarm"}[part_name]
        dof_inds.extend(Globals.robot.GetManipulator(manip_name).GetArmIndices())            
        trajs.append(traj)
    full_traj = np.concatenate(trajs, axis=1)
    Globals.robot.SetActiveDOFs(dof_inds)

    # make the trajectory slow enough for the simulation
    full_traj = ropesim.retime_traj(Globals.robot, dof_inds, full_traj)

    # in simulation mode, we must make sure to gradually move to the new starting position
    curr_vals = Globals.robot.GetActiveDOFValues()
    transition_traj = np.r_[[curr_vals], [full_traj[0]]]
    unwrap_in_place(transition_traj)
    transition_traj = ropesim.retime_traj(Globals.robot, dof_inds, transition_traj, max_cart_vel=.05)
    animate_traj.animate_traj(transition_traj, Globals.robot, restore=False, pause=interactive,
        callback=sim_callback, step_viewer=animate_speed)
    full_traj[0] = transition_traj[-1]
    unwrap_in_place(full_traj)

    animate_traj.animate_traj(full_traj, Globals.robot, restore=False, pause=interactive,
        callback=sim_callback, step_viewer=animate_speed)
    if Globals.viewer:
        Globals.viewer.Step()
    return True

def load_random_start_segment(demofile):
    start_keys = [k for k in demofile.keys() if k.startswith('demo') and k.endswith('00')]
    seg_name = random.choice(start_keys)
    return demofile[seg_name]['cloud_xyz']

def sample_rope_state(demofile, human_check=False, perturb_points=5, min_rad=0, max_rad=.15):
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

def replace_rope(new_rope):
    import bulletsimpy
    old_rope_nodes = Globals.sim.rope.GetControlPoints()
    if Globals.viewer:
        Globals.viewer.RemoveKinBody(Globals.env.GetKinBody('rope'))
    Globals.env.Remove(Globals.env.GetKinBody('rope'))
    Globals.sim.bt_env.Remove(Globals.sim.bt_env.GetObjectByName('rope'))
    Globals.sim.rope = bulletsimpy.CapsuleRope(Globals.sim.bt_env, 'rope', new_rope,
                                               Globals.sim.rope_params)
    return old_rope_nodes

def check_outfile(outfile):
    for k in outfile:        
        if not all(sub_g in outfile[k] for sub_g in ('action', 'cloud_xyz', 'knot', 'pred')):
            print "missing necessary groups"
            outfile.close()
            return False
        pred = int(outfile[k]['pred'][()])
        if pred != int(k) and pred != int(k) -1:            
            print "predecessors not correct", k, pred            
            outfile.close()
            return False
        knot = outfile[k]['knot'][()]
        action = outfile[k]['action'][()]
        if knot and not action.startswith('endstate'):
            print "end states labelled improperly"
            outfile.close()
            return False
    return True

def concat_datafiles(in_f1, in_f2, ofname):
    """ 
    assumes both files are opened in append mode
    puts the examples from of2 into of1
    """
    if not check_outfile(in_f1):
        in_f2.close()
        raise Exception, "input file 1 not formatted correctly"
    if not check_outfile(in_f2):
        in_f1.close()
        raise Exception, "input file 2 not formatted correctly " + str(in_f2)
    of = h5py.File(ofname, 'a')
    offset = len(of)
    for k, g in in_f1.iteritems():
        new_id = int(k) + offset  
        new_pred = str(int(g['pred'][()]) + offset)
        write_flush(of, [['action', g['action'][()]],
                         ['cloud_xyz', g['cloud_xyz'][:]],
                         ['knot', g['knot'][()]],
                         ['pred', new_pred]],
                    key = str(new_id))
    offset = len(of)
    for k, g in in_f2.iteritems():
        new_id = int(k) + offset  
        new_pred = str(int(g['pred'][()]) + offset)
        write_flush(of, [['action', g['action'][()]],
                         ['cloud_xyz', g['cloud_xyz'][:]],
                         ['knot', g['knot'][()]],
                         ['pred', new_pred]],
                    key = str(new_id))
    return check_outfile(of)    # return False if something isn't formatted right

def write_flush(outfile, items, key=None):
    if not key:
        key = str(len(outfile))
    g = outfile.create_group(key)
    for k, v in items:
        g[k] = v
    outfile.flush()

def h5_no_endstate_len(outfile):
    ctr = 0
    for k in outfile:
        if not outfile[k]['knot'][()]:
            ctr += 1
    print "num examples in file:\t", ctr
    return ctr

def remove_last_example(outfile):
    key = str(len(outfile) - 1)
    try:
        while True:
            ## will loop until we get something that is its own pred
            new_key = str(outfile[key]['pred'])
            del outfile[key]
            key = new_key
    except:
        key = str(len(outfile)-1)
        if not outfile[key]['knot']:
            raise Exception, "issue deleting examples, check your file"

def get_input(start_state, action_name, next_state, outfile, pred):
    print "d accepts and resamples rope"
    print "i ignores and resamples rope"
    print "r removes this entire example"
    print "you can C-c to quit safely"
    response = raw_input("Use this demonstration?[y/N/d/i/r]")
    resample = False
    success = False
    if response in ('R', 'r'):
        remove_last_example(outfile)
        resample = True
    elif response in ('I', 'i'):
        resample = True
    elif response in ('D', 'd'):
        resample = True
        # write the demonstration
        write_flush(outfile, 
                    items=[['cloud_xyz', start_state],
                           ['action', action_name],
                           ['knot', 0], # additional flag to tell if this is a knot
                           ['pred', pred]])
        # write the end state
        write_flush(outfile,
                    items = [['cloud_xyz', next_state],
                             ['action', 'endstate:' + action_name],
                             ['knot', 1],
                             ['pred', str(len(outfile)-1)]])
        success = True
    elif response in ('Y', 'y'):
        write_flush(outfile, 
                    items=[['cloud_xyz', start_state],
                           ['action', action_name],
                           ['knot', 0], # additional flag to tell if this is a knot
                           ['pred', pred]]) 
        success = True
    return (success, resample)    


def manual_select_demo(xyz, demofile, outfile, pred):
    start_rope_state = Globals.sim.rope.GetControlPoints()    
    ds_clouds = dict(zip(demofile.keys(), get_downsampled_clouds(demofile)))
    if args.parallel:
        ds_items = sorted(ds_clouds.items())
        costs = Parallel(n_jobs=-1,verbose=100)(delayed(registration_cost_cheap)(ds_cloud, xyz) for (s_name, ds_cloud) in ds_items)
        names_costs = [(ds_items[i][0], costs[i]) for i in range(len(ds_items))]
        costs = dict(names_costs)
    else:
        costs = {}
        for i, (seg_name, ds_cloud) in enumerate(ds_clouds.items()):
            costs[seg_name] = registration_cost_cheap(ds_cloud, xyz)
            sys.stdout.write("completed %i/%i\r"%(i+1, len(ds_clouds)))
            sys.stdout.flush()        
            sys.stdout.write('\n')
    best_keys = sorted(costs, key=costs.get)
    for seg_name in best_keys:
        sim_success = simulate_demo(xyz, demofile[seg_name], animate=True)
        new_xyz = Globals.sim.observe_cloud()
        if not sim_success: 
            replace_rope(start_rope_state)
            continue # no point in going further with this one
        (success, resample) = get_input(xyz, str(seg_name), new_xyz, outfile, pred)
        if resample:
            sample_rope_state(demofile)
            break
        elif success:
            break
        else:
            replace_rope(start_rope_state)
    if resample:
        # return the key for the next sample we'll see (so it is its own pred)
        return str(len(outfile))
    else:
        # return the key for the most recent addition
        return str(len(outfile)-1)
        


DS_SIZE = .025

def simulate_demo(new_xyz, seg_info, animate=False):
    Globals.robot.SetDOFValues(PR2_L_POSTURES["side"], Globals.robot.GetManipulator("leftarm").GetArmIndices())
    Globals.robot.SetDOFValues(mirror_arm_joints(PR2_L_POSTURES["side"]), Globals.robot.GetManipulator("rightarm").GetArmIndices())
    
    redprint("Generating end-effector trajectory")    
    
    handles = []
    old_xyz = np.squeeze(seg_info["cloud_xyz"])
    handles.append(Globals.env.plot3(old_xyz,5, (1,0,0)))
    handles.append(Globals.env.plot3(new_xyz,5, (0,0,1)))
    
    old_xyz = clouds.downsample(old_xyz, DS_SIZE)
    new_xyz = clouds.downsample(new_xyz, DS_SIZE)
    
    link_names = ["%s_gripper_tool_frame"%lr for lr in ('lr')]
    hmat_list = [(lr, seg_info[ln]['hmat']) for lr, ln in zip('lr', link_names)]
    lr2eetraj = warp_hmats(old_xyz, new_xyz, hmat_list)[0]

    miniseg_starts, miniseg_ends = split_trajectory_by_gripper(seg_info)    
    success = True
    print colorize.colorize("mini segments:", "red"), miniseg_starts, miniseg_ends
    for (i_miniseg, (i_start, i_end)) in enumerate(zip(miniseg_starts, miniseg_ends)):            

        ################################    
        redprint("Generating joint trajectory for part %i"%(i_miniseg))

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
            new_ee_traj = lr2eetraj[lr][i_start:i_end+1]          
            new_ee_traj_rs = resampling.interp_hmats(timesteps_rs, np.arange(len(new_ee_traj)), new_ee_traj)
            print "planning trajectory following"
            with util.suppress_stdout():
                new_joint_traj = planning.plan_follow_traj(Globals.robot, manip_name,
                                                           Globals.robot.GetLink(ee_link_name), new_ee_traj_rs,old_joint_traj_rs)
            part_name = {"l":"larm", "r":"rarm"}[lr]
            bodypart2traj[part_name] = new_joint_traj
            ################################    
            redprint("Executing joint trajectory for part %i using arms '%s'"%(i_miniseg, bodypart2traj.keys()))

        for lr in 'lr':
            gripper_open = binarize_gripper(seg_info["%s_gripper_joint"%lr][i_start])
            prev_gripper_open = binarize_gripper(seg_info["%s_gripper_joint"%lr][i_start-1]) if i_start != 0 else False
            if not set_gripper_maybesim(lr, gripper_open, prev_gripper_open):
                redprint("Grab %s failed" % lr)
                success = False

        if not success: break

        if len(bodypart2traj) > 0:
            success &= sim_traj_maybesim(bodypart2traj, animate=True)

        if not success: break

    Globals.sim.settle(animate=animate)
    Globals.robot.SetDOFValues(PR2_L_POSTURES["side"], Globals.robot.GetManipulator("leftarm").GetArmIndices())
    Globals.robot.SetDOFValues(mirror_arm_joints(PR2_L_POSTURES["side"]), Globals.robot.GetManipulator("rightarm").GetArmIndices())

    Globals.sim.release_rope('l')
    Globals.sim.release_rope('r')
    
    return success

def replace_rope(new_rope):
    import bulletsimpy
    old_rope_nodes = Globals.sim.rope.GetControlPoints()
    if Globals.viewer:
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

def load_fake_data_segment(demofile, fake_data_segment, fake_data_transform, set_robot_state=True):
    fake_seg = demofile[fake_data_segment]
    new_xyz = np.squeeze(fake_seg["cloud_xyz"])
    hmat = openravepy.matrixFromAxisAngle(fake_data_transform[3:6])
    hmat[:3,3] = fake_data_transform[0:3]
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

def reset_arms_to_side():
    Globals.robot.SetDOFValues(PR2_L_POSTURES["side"],
                               Globals.robot.GetManipulator("leftarm").GetArmIndices())
    Globals.robot.SetDOFValues(mirror_arm_joints(PR2_L_POSTURES["side"]),
                               Globals.robot.GetManipulator("rightarm").GetArmIndices())

###################

class Globals:
    robot = None
    env = None
    pr2 = None
    sim = None
    log = None
    viewer = None
    resample_rope = None

if __name__ == "__main__":
    """
    example command:
    ./do_task_eval.py data/weights/multi_quad_weights_10000.h5 --quad_features --animation=1
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('actionfile')
    parser.add_argument('outfile')
        
    parser.add_argument("--fake_data_segment",type=str, default='demo1-seg00')
    parser.add_argument("--fake_data_transform", type=float, nargs=6, metavar=("tx","ty","tz","rx","ry","rz"),
        default=[0,0,0,0,0,0], help="translation=(tx,ty,tz), axis-angle rotation=(rx,ry,rz)")
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument("--interactive",action="store_true")
    parser.add_argument("--log", type=str, default="", help="")
    parser.add_argument("--parallel", action="store_true")
    
    args = parser.parse_args()

    if args.random_seed is not None: np.random.seed(args.random_seed)

    trajoptpy.SetInteractive(args.interactive)

    if args.log:
        redprint("Writing log to file %s" % args.log)
        Globals.exec_log = task_execution.ExecutionLog(args.log)
        atexit.register(Globals.exec_log.close)
        Globals.exec_log(0, "main.args", args)

    Globals.env = openravepy.Environment()
    Globals.env.StopSimulation()
    Globals.env.Load("robots/pr2-beta-static.zae")
    Globals.robot = Globals.env.GetRobots()[0]

    actionfile = h5py.File(args.actionfile, 'r')
    outfile = h5py.File(args.outfile, 'a')
    
    init_rope_xyz, _ = load_fake_data_segment(actionfile, args.fake_data_segment, args.fake_data_transform) # this also sets the torso (torso_lift_joint) to the height in the data
    table_height = init_rope_xyz[:,2].mean() - .02
    table_xml = make_table_xml(translation=[1, 0, table_height], extents=[.85, .55, .01])
    Globals.env.LoadData(table_xml)
    Globals.sim = ropesim.Simulation(Globals.env, Globals.robot)
    # create rope from rope in data
    rope_nodes = rope_initialization.find_path_through_point_cloud(init_rope_xyz)
    Globals.sim.create(rope_nodes)
    # move arms to the side
    reset_arms_to_side()

    Globals.viewer = trajoptpy.GetViewer(Globals.env)
    print "move viewer to viewpoint that isn't stupid"
    print "then hit 'p' to continue"
    Globals.viewer.Idle()

    sample_rope_state(actionfile)

    #####################
    try:
        pred = str(len(outfile))
        while True:
            reset_arms_to_side()
            
            Globals.sim.settle()
            Globals.viewer.Step()
        
            xyz = Globals.sim.observe_cloud()
            pred = manual_select_demo(xyz, actionfile, outfile, pred)
    except KeyboardInterrupt:
        actionfile.close()
        h5_no_endstate_len(outfile)
        safe = check_outfile(outfile)
        if not safe:
            print args.outfile+" is not properly formatted, check it manually!!!!!"
