#!/usr/bin/env python

# Run this script with ./generate_holdout <demos_h5file> <output_filename>
#
# This script generates an h5 file containing rope starting states for the
# holdout set. Each starting state has its own group in the file, with a
# unique id ("0", "1", "2", "3", ...). Within each group is two keys,
# "rope_nodes" corresponding to the rope coords of the starting state, and
# "demo_id" corresponding to the demonstration starting state in the input
# <demos_file> (e.g. all.h5) that this starting state was perturbed from.

import argparse
usage="""
Generate holdout set with
./generate_holdout <demos_h5file> <output_filename>

Other options:
--human_check        # Asks for user confirmation for each holdout data item
--perturb_points 5   # Number of points perturbed from demonstration starting state
--min_rad 0          # Minimum perturbation 
--max_rad 0.15       # Maximum perturbation
--dataset_size 100   # Number of starting states to generate
"""
parser = argparse.ArgumentParser(usage=usage)
parser.add_argument("demos_h5file", type=str)
parser.add_argument("holdout_file", type=str)

parser.add_argument("--human_check", action="store_true")
parser.add_argument("--no_animation", action="store_true")
parser.add_argument("--perturb_points", type=int, default=5)
parser.add_argument("--min_rad", type=float, default=0)
parser.add_argument("--max_rad", type=float, default=0.15)
parser.add_argument("--dataset_size", type=int, default=100)

args = parser.parse_args()

from rapprentice import ropesim, rope_initialization
import openravepy, trajoptpy
import h5py, random, numpy as np

class Globals:
    env = None
    robot = None
    sim = None
    viewer = None
    animation = True

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
    """ % (translation[0], translation[1], translation[2],
           extents[0], extents[1], extents[2])
    return xml

def load_random_start_segment(demofile):
    start_keys = [k for k in demofile.keys()
                  if k.startswith('demo') and k.endswith('00')]
    seg_name = random.choice(start_keys)
    return (demofile[seg_name]['cloud_xyz'], seg_name)

def replace_rope(new_rope):
    import bulletsimpy

    if Globals.env.GetKinBody('rope') is None:
        Globals.sim.create(new_rope)
        return

    if Globals.animation:
        Globals.viewer.RemoveKinBody(Globals.env.GetKinBody('rope'))
    Globals.env.Remove(Globals.env.GetKinBody('rope'))
    Globals.sim.bt_env.Remove(Globals.sim.bt_env.GetObjectByName('rope'))
    Globals.sim.rope = bulletsimpy.CapsuleRope(Globals.sim.bt_env, 'rope',
                                               new_rope, Globals.sim.rope_params)

def sample_rope_state(demofile):
    success = False
    while not success:
        new_xyz, demo_id = load_random_start_segment(demofile)
        perturb_radius = random.uniform(args.min_rad, args.max_rad)
        rope_nodes = rope_initialization.find_path_through_point_cloud(
                         new_xyz,
                         perturb_peak_dist = perturb_radius,
                         num_perturb_points = args.perturb_points)

        replace_rope(rope_nodes)
        Globals.sim.settle()
        if Globals.animation:
            Globals.viewer.Step()
        if args.human_check:
            resp = raw_input("Use this start simulation?[Y/n]")
            success = resp not in ('N', 'n')
        else:
            success = True
        #new_xyz = Globals.sim.observe_cloud()

    return (rope_nodes, demo_id)

def save_example_action(rope_nodes, demo_id):
    with h5py.File(args.holdout_file, 'a') as f:
        if f.keys():
            "new key just adds one to the largest index"
            new_k = str(sorted([int(k) for k in f.keys()])[-1] + 1)
        else:
            "new key starts at 0"
            new_k = '0'
        f.create_group(new_k)
        f[new_k]['rope_nodes'] = rope_nodes
        f[new_k]['demo_id'] = str(demo_id)
        f.close()

PR2_L_POSTURES = dict(
    untucked = [0.4,  1.0,   0.0,  -2.05,  0.0,  -0.1,  0.0],
    tucked = [0.06, 1.25, 1.79, -1.68, -1.73, -0.10, -0.09],
    up = [ 0.33, -0.35,  2.59, -0.15,  0.59, -1.41, -0.27],
    side = [  1.832,  -0.332,   1.011,  -1.437,   1.1  ,  -2.106,  3.074]
)

def mirror_arm_joints(x):
    "mirror image of joints (r->l or l->r)"
    return np.r_[-x[0],x[1],-x[2],x[3],-x[4],x[5],-x[6]]

def main():
    demofile = h5py.File(args.demos_h5file, 'r')
    trajoptpy.SetInteractive(False)

    if args.no_animation:
        Globals.animation = False

    Globals.env = openravepy.Environment()
    Globals.env.StopSimulation()
    Globals.env.Load("robots/pr2-beta-static.zae")
    Globals.robot = Globals.env.GetRobots()[0]

    Globals.robot.SetDOFValues(PR2_L_POSTURES["side"], Globals.robot.GetManipulator("leftarm").GetArmIndices())
    Globals.robot.SetDOFValues(mirror_arm_joints(PR2_L_POSTURES["side"]), Globals.robot.GetManipulator("rightarm").GetArmIndices())
    
    init_rope_xyz, _ = load_random_start_segment(demofile)
    table_height = init_rope_xyz[:,2].mean() - .02
    table_xml = make_table_xml(translation=[1, 0, table_height], extents=[.85, .55, .01])
    Globals.env.LoadData(table_xml)
    Globals.sim = ropesim.Simulation(Globals.env, Globals.robot)

    if Globals.animation:
        Globals.viewer = trajoptpy.GetViewer(Globals.env)
        Globals.viewer.Idle()

    for i in range(0, args.dataset_size):
        print "State ", i, " of ", args.dataset_size
        rope_nodes, demo_id = sample_rope_state(demofile)
        save_example_action(rope_nodes, demo_id)

if __name__ == "__main__":
    main()
