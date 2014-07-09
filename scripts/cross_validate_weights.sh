#!/bin/bash

# python build.py  base bellman full --C .00001 .0001 .001 .01 .1 1 10 100 1000
# python build.py  mul bellman full --C .00001 .0001 .001 .01 .1 1 10 100 1000
# python build.py  mul_s bellman full --C .00001 .0001 .001 .01 .1 1 10 100 1000
# python build.py  mul_quad bellman full --C .00001 .0001 .001 .01 .1 1 10 100 1000
# python build.py  landmark bellman full --C .00001 .0001 .001 .01 .1 1 10 100 1000

# base evaluations
python do_task_eval.py --resultfile ../data/evals/jul_6_base_0.1_c\=1e-05_bellman.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 base --weightfile ../data/weights/labels_Jul_3_0.1_base_c\=1e-05_bellman.h5
python do_task_eval.py --resultfile ../data/evals/jul_6_base_0.1_c\=0.0001_bellman.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 base --weightfile ../data/weights/labels_Jul_3_0.1_base_c\=0.0001_bellman.h5
python do_task_eval.py --resultfile ../data/evals/jul_6_base_0.1_c\=0.001_bellman.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 base --weightfile ../data/weights/labels_Jul_3_0.1_base_c\=0.001_bellman.h5
python do_task_eval.py --resultfile ../data/evals/jul_6_base_0.1_c\=0.01_bellman.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 base --weightfile ../data/weights/labels_Jul_3_0.1_base_c\=0.01_bellman.h5
python do_task_eval.py --resultfile ../data/evals/jul_6_base_0.1_c\=0.1_bellman.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 base --weightfile ../data/weights/labels_Jul_3_0.1_base_c\=0.1_bellman.h5
python do_task_eval.py --resultfile ../data/evals/jul_6_base_0.1_c\=1.0_bellman.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 base --weightfile ../data/weights/labels_Jul_3_0.1_base_c\=1.0_bellman.h5
python do_task_eval.py --resultfile ../data/evals/jul_6_base_0.1_c\=10.0_bellman.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 base --weightfile ../data/weights/labels_Jul_3_0.1_base_c\=10.0_bellman.h5
python do_task_eval.py --resultfile ../data/evals/jul_6_base_0.1_c\=100.0_bellman.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 base --weightfile ../data/weights/labels_Jul_3_0.1_base_c\=100.0_bellman.h5
python do_task_eval.py --resultfile ../data/evals/jul_6_base_0.1_c\=1000.0_bellman.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 base --weightfile ../data/weights/labels_Jul_3_0.1_base_c\=1000.0_bellman.h5

# mul features
python do_task_eval.py --resultfile ../data/evals/jul_6_mul_0.1_c\=1e-05_bellman.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 mul --weightfile ../data/weights/labels_Jul_3_0.1_mul_c\=1e-05_bellman.h5
python do_task_eval.py --resultfile ../data/evals/jul_6_mul_0.1_c\=0.0001_bellman.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 mul --weightfile ../data/weights/labels_Jul_3_0.1_mul_c\=0.0001_bellman.h5
python do_task_eval.py --resultfile ../data/evals/jul_6_mul_0.1_c\=0.001_bellman.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 mul --weightfile ../data/weights/labels_Jul_3_0.1_mul_c\=0.001_bellman.h5
python do_task_eval.py --resultfile ../data/evals/jul_6_mul_0.1_c\=0.01_bellman.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 mul --weightfile ../data/weights/labels_Jul_3_0.1_mul_c\=0.01_bellman.h5
python do_task_eval.py --resultfile ../data/evals/jul_6_mul_0.1_c\=0.1_bellman.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 mul --weightfile ../data/weights/labels_Jul_3_0.1_mul_c\=0.1_bellman.h5
python do_task_eval.py --resultfile ../data/evals/jul_6_mul_0.1_c\=1.0_bellman.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 mul --weightfile ../data/weights/labels_Jul_3_0.1_mul_c\=1.0_bellman.h5
python do_task_eval.py --resultfile ../data/evals/jul_6_mul_0.1_c\=10.0_bellman.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 mul --weightfile ../data/weights/labels_Jul_3_0.1_mul_c\=10.0_bellman.h5
python do_task_eval.py --resultfile ../data/evals/jul_6_mul_0.1_c\=100.0_bellman.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 mul --weightfile ../data/weights/labels_Jul_3_0.1_mul_c\=100.0_bellman.h5
python do_task_eval.py --resultfile ../data/evals/jul_6_mul_0.1_c\=1000.0_bellman.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 mul --weightfile ../data/weights/labels_Jul_3_0.1_mul_c\=1000.0_bellman.h5

# mul_s features
python do_task_eval.py --resultfile ../data/evals/jul_6_mul_s_0.1_c\=1e-05_bellman.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 mul_s --weightfile ../data/weights/labels_Jul_3_0.1_mul_s_c\=1e-05_bellman.h5
python do_task_eval.py --resultfile ../data/evals/jul_6_mul_s_0.1_c\=0.0001_bellman.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 mul_s --weightfile ../data/weights/labels_Jul_3_0.1_mul_s_c\=0.0001_bellman.h5
python do_task_eval.py --resultfile ../data/evals/jul_6_mul_s_0.1_c\=0.001_bellman.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 mul_s --weightfile ../data/weights/labels_Jul_3_0.1_mul_s_c\=0.001_bellman.h5
python do_task_eval.py --resultfile ../data/evals/jul_6_mul_s_0.1_c\=0.01_bellman.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 mul_s --weightfile ../data/weights/labels_Jul_3_0.1_mul_s_c\=0.01_bellman.h5
python do_task_eval.py --resultfile ../data/evals/jul_6_mul_s_0.1_c\=0.1_bellman.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 mul_s --weightfile ../data/weights/labels_Jul_3_0.1_mul_s_c\=0.1_bellman.h5
python do_task_eval.py --resultfile ../data/evals/jul_6_mul_s_0.1_c\=1.0_bellman.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 mul_s --weightfile ../data/weights/labels_Jul_3_0.1_mul_s_c\=1.0_bellman.h5
python do_task_eval.py --resultfile ../data/evals/jul_6_mul_s_0.1_c\=10.0_bellman.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 mul_s --weightfile ../data/weights/labels_Jul_3_0.1_mul_s_c\=10.0_bellman.h5
python do_task_eval.py --resultfile ../data/evals/jul_6_mul_s_0.1_c\=100.0_bellman.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 mul_s --weightfile ../data/weights/labels_Jul_3_0.1_mul_s_c\=100.0_bellman.h5
python do_task_eval.py --resultfile ../data/evals/jul_6_mul_s_0.1_c\=1000.0_bellman.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 mul_s --weightfile ../data/weights/labels_Jul_3_0.1_mul_s_c\=1000.0_bellman.h5

# mul_quad features
python do_task_eval.py --resultfile ../data/evals/jul_6_mul_quad_0.1_c\=1e-05_bellman.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 mul_quad --weightfile ../data/weights/labels_Jul_3_0.1_mul_quad_c\=1e-05_bellman.h5
python do_task_eval.py --resultfile ../data/evals/jul_6_mul_quad_0.1_c\=0.0001_bellman.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 mul_quad --weightfile ../data/weights/labels_Jul_3_0.1_mul_quad_c\=0.0001_bellman.h5
python do_task_eval.py --resultfile ../data/evals/jul_6_mul_quad_0.1_c\=0.001_bellman.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 mul_quad --weightfile ../data/weights/labels_Jul_3_0.1_mul_quad_c\=0.001_bellman.h5
python do_task_eval.py --resultfile ../data/evals/jul_6_mul_quad_0.1_c\=0.01_bellman.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 mul_quad --weightfile ../data/weights/labels_Jul_3_0.1_mul_quad_c\=0.01_bellman.h5
python do_task_eval.py --resultfile ../data/evals/jul_6_mul_quad_0.1_c\=0.1_bellman.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 mul_quad --weightfile ../data/weights/labels_Jul_3_0.1_mul_quad_c\=0.1_bellman.h5
python do_task_eval.py --resultfile ../data/evals/jul_6_mul_quad_0.1_c\=1.0_bellman.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 mul_quad --weightfile ../data/weights/labels_Jul_3_0.1_mul_quad_c\=1.0_bellman.h5
python do_task_eval.py --resultfile ../data/evals/jul_6_mul_quad_0.1_c\=10.0_bellman.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 mul_quad --weightfile ../data/weights/labels_Jul_3_0.1_mul_quad_c\=10.0_bellman.h5
python do_task_eval.py --resultfile ../data/evals/jul_6_mul_quad_0.1_c\=100.0_bellman.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 mul_quad --weightfile ../data/weights/labels_Jul_3_0.1_mul_quad_c\=100.0_bellman.h5
python do_task_eval.py --resultfile ../data/evals/jul_6_mul_quad_0.1_c\=1000.0_bellman.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 mul_quad --weightfile ../data/weights/labels_Jul_3_0.1_mul_quad_c\=1000.0_bellman.h5

# landmark evaluations
python do_task_eval.py --resultfile ../data/evals/jul_6_landmark_0.1_c\=1e-05_bellman.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 landmark --weightfile ../data/weights/labels_Jul_3_0.1_landmark_c\=1e-05_bellman.h5
python do_task_eval.py --resultfile ../data/evals/jul_6_landmark_0.1_c\=0.0001_bellman.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 landmark --weightfile ../data/weights/labels_Jul_3_0.1_landmark_c\=0.0001_bellman.h5
python do_task_eval.py --resultfile ../data/evals/jul_6_landmark_0.1_c\=0.001_bellman.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 landmark --weightfile ../data/weights/labels_Jul_3_0.1_landmark_c\=0.001_bellman.h5
python do_task_eval.py --resultfile ../data/evals/jul_6_landmark_0.1_c\=0.01_bellman.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 landmark --weightfile ../data/weights/labels_Jul_3_0.1_landmark_c\=0.01_bellman.h5
python do_task_eval.py --resultfile ../data/evals/jul_6_landmark_0.1_c\=0.1_bellman.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 landmark --weightfile ../data/weights/labels_Jul_3_0.1_landmark_c\=0.1_bellman.h5
python do_task_eval.py --resultfile ../data/evals/jul_6_landmark_0.1_c\=1.0_bellman.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 landmark --weightfile ../data/weights/labels_Jul_3_0.1_landmark_c\=1.0_bellman.h5
python do_task_eval.py --resultfile ../data/evals/jul_6_landmark_0.1_c\=10.0_bellman.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 landmark --weightfile ../data/weights/labels_Jul_3_0.1_landmark_c\=10.0_bellman.h5
python do_task_eval.py --resultfile ../data/evals/jul_6_landmark_0.1_c\=100.0_bellman.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 landmark --weightfile ../data/weights/labels_Jul_3_0.1_landmark_c\=100.0_bellman.h5
python do_task_eval.py --resultfile ../data/evals/jul_6_landmark_0.1_c\=1000.0_bellman.h5 --i_end 100 --animation 0 eval ../data/misc/actions.h5 ../data/misc/Jul_3_0.1_test.h5 landmark --weightfile ../data/weights/labels_Jul_3_0.1_landmark_c\=1000.0_bellman.h5

python auto_performance.py ../data/evals/jul_8_bellman_evals.h5