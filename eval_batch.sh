#!/bin/bash

#./rope_qlearn.py build-model data/all_constraints.h5 data/single_bias_model.mps
#./rope_qlearn.py --quad_features build-model data/all_constraints_quadratic.h5 data/single_quad_model.mps
#./rope_qlearn.py --multi_slack build-model data/all_constraints.h5 data/multi_bias_model.mps
#./rope_qlearn.py --multi_slack --quad_features build-model data/all_constraints_quadratic.h5 data/multi_quad_model.mps
#./rope_qlearn.py --sc_features build-model data/sc_features.h5 data/single_sc_model.mps
#./rope_qlearn.py --multi_slack --sc_features build-model data/sc_features.h5 data/multi_sc_model.mps
#./rope_qlearn.py --multi_slack --rope_dist_features build-model data/constraints/ropedist_constraints.h5 data/models/multi_ropedist_model.mps
for C in 0.1 0.5 1 5 10 50 100 500 1000 5000 10000
do    
    echo ${C}
    #./rope_qlearn.py --C ${C} optimize-model data/single_bias_model.mps data/single_bias_weights_${C}.h5 
    #./rope_qlearn.py --C ${C} --quad_features optimize-model data/single_quad_model.mps data/single_quad_weights_${C}.h5 
    #./rope_qlearn.py --C ${C} --multi_slack optimize-model data/multi_bias_model.mps data/multi_bias_weights_${C}.h5
    #./rope_qlearn.py --C ${C} --multi_slack --quad_features optimize-model data/multi_quad_model.mps data/multi_quad_weights_${C}.h5 
    #./rope_qlearn.py --C ${C} --sc_features optimize-model data/single_sc_model.mps data/single_sc_weights_${C}.h5 
    #./rope_qlearn.py --C ${C} --multi_slack --sc_features optimize-model data/multi_sc_model.mps data/multi_sc_weights_${C}.h5 
    ./rope_qlearn.py --C ${C} --multi_slack --rope_dist_features --save_memory optimize-model data/models/multi_ropedist_model.mps data/weights/multi_ropedist_weights_${C}.h5 
done
