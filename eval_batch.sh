#!/bin/bash

#./rope_qlearn.py build-model data/all_constraints.h5 data/single_bias_model.mps
#./rope_qlearn.py --quad_features build-model data/all_constraints_quadratic.h5 data/single_quad_model.mps
#./rope_qlearn.py --multi_slack build-model data/all_constraints.h5 data/multi_bias_model.mps
#./rope_qlearn.py --multi_slack --quad_features build-model data/all_constraints_quadratic.h5 data/multi_quad_model.mps
for C in 0.1 0.5 1 5 10 50 100 500 1000 5000 10000
do    
    echo ${C}
    ./rope_qlearn.py --C ${C} optimize-model data/single_bias_model.mps data/single_bias_weights_${C}.h5 
    ./rope_qlearn.py --C ${C} --quad_features optimize-model data/single_quad_model.mps data/single_quad_weights_${C}.h5 
    ./rope_qlearn.py --C ${C} --multi_slack optimize-model data/multi_bias_model.mps data/multi_bias_weights_${C}.h5
    ./rope_qlearn.py --C ${C} --multi_slack --quad_features optimize-model data/multi_quad_model.mps data/multi_quad_weights_${C}.h5 
done
