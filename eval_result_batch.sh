#!/bin/bash

for C in 0.1 0.5 1 5 10 50 100 500 1000 5000 10000
do    
    echo $1 ${C}

    mkdir data/eval/$1_${C}
    if echo $1 | grep -q "quad"; then
        ./do_task_eval.py --quad_features data/$1_weights_${C}.h5 data/eval/$1_${C}/holdout_result.h5
    else
        ./do_task_eval.py data/$1_weights_${C}.h5 data/eval/$1_${C}/holdout_result.h5
    fi

    mkdir data/eval/$1_${C}/images
    ./holdout_result.py data/eval/$1_${C}/holdout_result.h5 data/eval/$1_${C}/images
done
