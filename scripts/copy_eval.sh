#!/bin/bash
mkdir data/no_outlier_eval/multi_bias_$1
mkdir data/no_outlier_eval/multi_bias_$1/images
scp alex@rll5:~/rll/reinforcement-lfd/feature-dist/jobs/eval/multi_bias_500_result_no_outlier.h5 data/no_outlier_eval/multi_bias_$1/result.h5
./holdout_result.py data/no_outlier_eval/multi_bias_$1/result.h5 data/no_outlier_eval/multi_bias_$1/images
./eval_holdout_performance.py data/no_outlier_eval/multi_bias_$1/images data/no_outlier_eval/multi_bias_$1/result.out