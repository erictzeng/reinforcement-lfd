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
from knot_classifier import isKnot as is_knot
from do_task_eval import Globals, load_fake_data_segment, make_table_xml, reset_arms_to_side, redprint, replace_rope, simulate_demo_traj
import os, numpy as np, h5py
from numpy import asarray
import atexit
import importlib
from itertools import combinations
import IPython as ipy
import random

if __name__ == "__main__":
    """
    example command:
    ./do_task_eval.py data/weights/multi_quad_weights_10000.h5 --quad_features --animation=1
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('actionfile', nargs='?', default='data/misc/actions.h5')
    parser.add_argument('holdoutfile', nargs='?', default='data/misc/evaluation_set_Jan24.h5')
    parser.add_argument("resultfile", type=str)

    parser.add_argument("--animation", type=int, default=0)
    parser.add_argument("--i_start", type=int, default=-1)
    parser.add_argument("--i_end", type=int, default=-1)

    parser.add_argument("--tasks", nargs='+', type=int)
    parser.add_argument("--taskfile", type=str)
    
    parser.add_argument("--fake_data_segment",type=str, default='demo1-seg00')
    parser.add_argument("--fake_data_transform", type=float, nargs=6, metavar=("tx","ty","tz","rx","ry","rz"),
        default=[0,0,0,0,0,0], help="translation=(tx,ty,tz), axis-angle rotation=(rx,ry,rz)")
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument("--interactive",action="store_true")
    parser.add_argument("--log", type=str, default="", help="")
    
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

    if args.animation:
        Globals.viewer = trajoptpy.GetViewer(Globals.env)
        print "move viewer to viewpoint that isn't stupid"
        print "then hit 'p' to continue"
        Globals.viewer.Idle()

    holdoutfile = h5py.File(args.holdoutfile, 'r')
    resultfile = h5py.File(args.resultfile, 'r')
    
    unique_id = 0
    def get_unique_id():
        global unique_id
        unique_id += 1
        return unique_id-1

    tasks = [] if args.tasks is None else args.tasks
    if args.taskfile is not None:
        file = open(args.taskfile, 'r')
        for line in file.xreadlines():
            tasks.append(int(line[5:-1]))
    if args.i_start != -1 and args.i_end != -1:
        tasks = range(args.i_start, args.i_end)
        
    num_knots = 0
    num_not_knots = 0
    for i_task, demo_id_rope_nodes in (holdoutfile.iteritems() if not tasks else [(unicode(t),holdoutfile[unicode(t)]) for t in tasks]):
        reset_arms_to_side()

        redprint("Replace rope")
        rope_nodes = demo_id_rope_nodes["rope_nodes"][:]
        replace_rope(rope_nodes)
        Globals.sim.settle()
        if args.animation:
            Globals.viewer.Step()
        
        for i_step in range(len(resultfile[i_task])):
            print "task %s step %i" % (i_task, i_step)

            reset_arms_to_side()

            redprint("Observe point cloud")
            new_xyz = Globals.sim.observe_cloud()
            
            rope_nodes = resultfile[i_task][str(i_step)]['rope_nodes'][()]
            best_action = resultfile[i_task][str(i_step)]['best_action'][()]
            trajs_g = resultfile[i_task][str(i_step)]['trajs']
            trajs = []
            for i_traj in range(len(trajs_g)):
                trajs.append({})
                for (bodypart, bodyparttraj) in trajs_g[str(i_traj)].iteritems():
                    trajs[i_traj][bodypart] = bodyparttraj[()]
            q_values = resultfile[i_task][str(i_step)]['values'][()]

            simulate_demo_traj(new_xyz, actionfile[best_action], trajs, animate=args.animation)
            
            if is_knot(Globals.sim.rope.GetControlPoints()):
                num_knots += 1
            else:
                num_not_knots += 1
                
    print "success rate is", float(num_knots) / (num_knots + num_not_knots)
