import argparse
import numpy as np
import h5py
import os
import sys
import time

import do_task_eval as task
import rope_qlearn as qlearn

def initialize_sim(actionfile):
    task.Globals.env = task.openravepy.Environment()
    task.Globals.env.StopSimulation()
    task.Globals.env.Load("robots/pr2-beta-static.zae")
    task.Globals.robot = task.Globals.env.GetRobots()[0]
    init_rope_xyz, _ = task.load_fake_data_segment(actionfile, 'demo1-seg00', [0]*6)
    table_height = init_rope_xyz[:,2].mean() - .02
    table_xml = task.make_table_xml(translation=[1, 0, table_height], extents=[.85, .55, .01])
    task.Globals.env.LoadData(table_xml)
    task.Globals.sim = task.ropesim.Simulation(task.Globals.env, task.Globals.robot)
    rope_nodes = task.rope_initialization.find_path_through_point_cloud(init_rope_xyz)
    task.Globals.sim.create(rope_nodes)
    task.reset_arms_to_side()

def get_rope_nodes(cloud):
    return task.rope_initialization.find_path_through_point_cloud(cloud)

def wait():
    os.system("lsof | awk '{{ print $2; }}' | sort -rn | uniq -c | sort -rn | grep {}".format(os.getpid()))
    time.sleep(5)

def try_action_on_state(state, action, animate=False):
    task.reset_arms_to_side()
    task.replace_rope(state)
    task.Globals.sim.settle()
    success = task.simulate_demo(state, action, animate=animate)
    if not success:
        return None
    else:
        return task.Globals.sim.observe_cloud()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('actionfile', nargs='?', default='data/misc/actions.h5')
    parser.add_argument('examplesfile', nargs='?', default='data/misc/auto_labeled_examples.h5')
    parser.add_argument('start', nargs='?', type=int, default=0)
    parser.add_argument('end', nargs='?', type=int, default=-1)
    parser.add_argument('--animate', action='store_true', default=False)
    args = parser.parse_args()

    actionfile = h5py.File(args.actionfile, 'r')
    examplesfile = h5py.File(args.examplesfile, 'w')
    initialize_sim(actionfile)

    if args.end < 0:
        args.end = float('inf')

    # awesome simulation stuff
    example_i = 0
    for ind, (action_name, action) in enumerate(actionfile.iteritems()):
        if ind < args.start or ind >= args.end:
            continue
        start = get_rope_nodes(np.squeeze(action['cloud_xyz']))
        target = try_action_on_state(start, action, animate=args.animate)
        if target is None: # this should never happen
            print >>sys.stderr, 'Action {} failed on its own state!'.format(action_name)
            continue
        best_action, best_cost = None, float('inf')
        for other_action_name, other_action in actionfile.iteritems():
            if action_name == other_action_name:
                continue
            result = try_action_on_state(start, other_action, animate=args.animate)
            if result is None: # this will probably happen a lot
                continue
            _, _, _, _, cost = qlearn.registration_cost(result, target)
            if cost < best_cost:
                best_action, best_cost = other_action_name, cost
        if best_action is None: # this should never happen either
            print >>sys.stderr, 'Action {} had no working examples!'.format(action_name)
            continue
        index = str(example_i)
        group = examplesfile.create_group(index)
        group['action'] = str(best_action)
        group['orig_action'] = str(action_name)
        group['cloud_xyz'] = start
        example_i += 1
