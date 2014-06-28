#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Error. Usage: ./eval_result_single.sh result_dir weight_file"
    exit 1
fi

result_dir=$1
weight_file=$2

echo "Using result directory $result_dir and weight file $weight_file"

mkdir $result_dir
if echo $result_dir | grep -q "quad"; then
    ./do_task_eval.py $weight_file --resultfile=$result_dir/holdout_result.h5 --quad_features
elif echo $result_dir | grep -q "sc"; then
    ./do_task_eval.py $weight_file --resultfile=$result_dir/holdout_result.h5 --sc_features
elif echo $result_dir | grep -q "ropedist"; then
    ./do_task_eval.py $weight_file --resultfile=$result_dir/holdout_result.h5 --rope_dist_features
else
    ./do_task_eval.py $weight_file --resultfile=$result_dir/holdout_result.h5
fi

mkdir $result_dir/images
./holdout_result.py $result_dir/holdout_result.h5 $result_dir/images

