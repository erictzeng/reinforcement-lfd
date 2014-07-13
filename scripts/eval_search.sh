#!/bin/bash

## weightfiles --weightfile ../data/weights/labels_Jul_3_0.1_landmark_c\=100.0_bellman.h5
##                           ../data/weights/labels_Jul_3_0.1_mul_quad_c\=1000.0_bellman.h5
##                           ../data/weights/labels_Jul_3_0.1_mul_c\=100.0_bellman.h5

python do_task_eval.py --width 3 --depth 3 --resultfile ../data/evals/search_eval_landmark_depth\=3_width\=3.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 landmark --weightfile ../data/weights/labels_Jul_3_0.1_landmark_c\=100.0_bellman.h5
# python do_task_eval.py --width 3 --depth 3 --resultfile ../data/evals/search_eval_mul_quad_depth\=3_width\=3.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 mul_quad --weightfile ../data/weights/labels_Jul_3_0.1_mul_quad_c\=1000.0_bellman.h5
python do_task_eval.py --width 3 --depth 3 --resultfile ../data/evals/search_eval_mul_depth\=3_width\=3.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 mul --weightfile ../data/weights/labels_Jul_3_0.1_mul_c\=100.0_bellman.h5


python do_task_eval.py --width 5 --depth 3 --resultfile ../data/evals/search_eval_landmark_depth\=3_width\=5.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 landmark --weightfile ../data/weights/labels_Jul_3_0.1_landmark_c\=100.0_bellman.h5
# python do_task_eval.py --width 5 --depth 3 --resultfile ../data/evals/search_eval_mul_quad_depth\=3_width\=5.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 mul_quad --weightfile ../data/weights/labels_Jul_3_0.1_mul_quad_c\=1000.0_bellman.h5
python do_task_eval.py --width 5 --depth 3 --resultfile ../data/evals/search_eval_mul_depth\=3_width\=5.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 mul --weightfile ../data/weights/labels_Jul_3_0.1_mul_c\=100.0_bellman.h5


python do_task_eval.py --width 10 --depth 2 --resultfile ../data/evals/search_eval_landmark_depth\=2_width\=10.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 landmark --weightfile ../data/weights/labels_Jul_3_0.1_landmark_c\=100.0_bellman.h5
# python do_task_eval.py --width 10 --depth 2 --resultfile ../data/evals/search_eval_mul_quad_depth\=2_width\=10.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 mul_quad --weightfile ../data/weights/labels_Jul_3_0.1_mul_quad_c\=1000.0_bellman.h5
python do_task_eval.py --width 10 --depth 2 --resultfile ../data/evals/search_eval_mul_depth\=2_width\=10.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 mul --weightfile ../data/weights/labels_Jul_3_0.1_mul_c\=100.0_bellman.h5

python ../scripts/tmp_auto_perf.py