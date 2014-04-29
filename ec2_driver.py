import rope_qlearn as qlearn

import argparse
import os.path

DATADIR = '/home/ubuntu/reinforcement-lfd/data'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('constraintfile')
    parser.add_argument('start', nargs='?', type=int, default=0)
    parser.add_argument('end', nargs='?', type=int, default=-1)
    parser.add_argument('--demofile', nargs='?', default=os.path.join(DATADIR, 'expert_demos.h5'))
    parser.add_argument('--actionfile', nargs='?', default=os.path.join(DATADIR, 'all.h5'))
    parser.add_argument("--quad_features", action="store_true")
    parser.add_argument("--sc_features", action="store_true")
    parser.add_argument("--rope_dist_features", action="store_true")
    parser.add_argument('--C', '-c', type=float, default=1)
    parser.add_argument("--save_memory", action="store_true")
    parser.add_argument("--gripper_weighting", action="store_true")
    args = parser.parse_args()
    act_set = qlearn.ActionSet(args.actionfile, gripper_weighting=args.gripper_weighting)
    feature_fn, margin_fn, num_features = qlearn.select_feature_fn(args, act_set)
    qlearn.compute_constraints_no_model(feature_fn, margin_fn, act_set.actions, args.demofile,
                                        args.constraintfile, start=args.start, end=args.end,
                                        verbose=True)
