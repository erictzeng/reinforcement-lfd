#!/bin/bash

M=20 # Number of iterations of DAgger to run
N=50 # Number of start states to sample at each step

for I in {1..${M}}
do
    echo ${I}

    # Sample N complete trajectories, for five time steps or until knot
    ./generate_holdout.py --no_animation --perturb_points 5 --min_rad 0 --max_rad 0.15 --dataset_size ${N} data/misc/actions.h5 data/dagger/startstates_iter${I}.h5
    ./do_task_eval.py --resultfile data/dagger/sampled_states_iter${I}.h5 --quad_landmark_features --landmark_features data/misc/landmarks/landmarks_70.h5 --rbf data/misc/actions.h5 data/dagger/startstates_iter${I}.h5 data/dagger/weights_iter${I}.h5

    # TODO: Label the sampled states

    # TODO: Generate constraints for these new labels, and save to file

    # TODO: Load saved model and add new constraints

    # TODO: Optimize the new model and generate new weights
done
