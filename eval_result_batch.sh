#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Error. Usage: ./eval_result_batch.sh experiment_name"
    exit 1
fi

#for C in 0.1 1 10 100 1000 10000
for C in 0.5 5 50 500 5000
#for C in 0.1 0.5 1 5 10 50 100 500 1000 5000 10000
do    
    echo "Evaluating experiment $1 with C = ${C}"

    result_dir=data/results/$1_${C}
    weight_file=data/weights/$1_weights_${C}.h5

    ./eval_result_single.sh $result_dir $weight_file

done

