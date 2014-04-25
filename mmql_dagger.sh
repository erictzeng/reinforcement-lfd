#!/bin/bash

# Inputs:
#     1. Input basename
#     2. Output basename
#     3. Number of states to sample at each step
#
# Note: There should exist a model file called {input_basename}_model.mps
#       There should also exist a weights file called {input_basename}_weights.h5

[ $# -eq 4 ] && { echo "Usage: $0 IN_PREFIX OUT_PREFIX N"; exit 1; }

IN_PREFIX=${1}
OUT_PREFIX=${2}
N=${3}

# Sample N complete trajectories, for five time steps or until knot
./generate_holdout.py --no_animation --perturb_points 5 --min_rad 0 --max_rad 0.15 --dataset_size ${N} data/misc/actions.h5 ${OUT_PREFIX}_startstates_${N}.h5
./do_task_eval.py --resultfile ${OUT_PREFIX}_sampled_states.h5 data/misc/actions.h5 ${OUT_PREFIX}_startstates_${N}.h5 eval --quad_landmark_features --landmark_features data/misc/landmarks/landmarks_70.h5 --rbf ${IN_PREFIX}_weights.h5

# Label the sampled states; then remove trajectories that end in deadends
./do_task_label.py --dagger_states_file ${OUT_PREFIX}_sampled_states.h5 data/misc/actions.h5 ${OUT_PREFIX}_labeled_states.h5
./filter_labeled_examples.py --remove_deadend_traj ${OUT_PREFIX}_labeled_states.h5 ${OUT_PREFIX}_labeled_states_nodeadend.h5

# Generate constraints for these new labels, and save to file
./rope_qlearn.py --quad_landmark_features --landmark_features data/misc/landmarks/landmarks_70.h5 --rbf build-constraints-no-model ${OUT_PREFIX}_labeled_states_nodeadend.h5 ${OUT_PREFIX}_constraints.h5 data/misc/actions.h5 bellman

# Load saved model and add new constraints
./rope_qlearn.py --quad_landmark_features --landmark_features data/misc/landmarks/landmarks_70.h5 --rbf build-model-merge ${OUT_PREFIX}_constraints.h5 ${IN_PREFIX}_model.mps ${OUT_PREFIX}_model.mps data/misc/actions.h5 bellman

# Optimize the new model and generate new weights
./rope_qlearn.py --quad_landmark_features --landmark_features data/misc/landmarks/landmarks_70.h5 --rbf optimize-model ${IN_PREFIX}_model.mps --C 2 --D 1 --F 1 ${OUT_PREFIX}_weights.h5 data/misc/actions.h5 bellman
