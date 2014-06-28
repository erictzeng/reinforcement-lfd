import openravepy
import numpy as np
import util
from rapprentice import planning

PR2_L_POSTURES = dict(
    untucked = [0.4,  1.0,   0.0,  -2.05,  0.0,  -0.1,  0.0],
    tucked = [0.06, 1.25, 1.79, -1.68, -1.73, -0.10, -0.09],
    up = [ 0.33, -0.35,  2.59, -0.15,  0.59, -1.41, -0.27],
    side = [  1.832,  -0.332,   1.011,  -1.437,   1.1  ,  -2.106,  3.074]
)

def initialize_lite_sim():
    env = openravepy.Environment()
    env.StopSimulation()
    env.Load("robots/pr2-beta-static.zae")
    robot = env.GetRobots()[0]
    return env, robot

def joint_trajs(action, actionfile):
    return dict(zip('lr', [actionfile[action][m_name][:] for m_name in ('leftarm', 'rightarm')]))

def mirror_arm_joints(x):
    "mirror image of joints (r->l or l->r)"
    return np.r_[-x[0],x[1],-x[2],x[3],-x[4],x[5],-x[6]]

def reset_arms_to_side(robot):
    robot.SetDOFValues(PR2_L_POSTURES["side"],
                               robot.GetManipulator("leftarm").GetArmIndices())
    robot.SetDOFValues(mirror_arm_joints(PR2_L_POSTURES["side"]),
                               robot.GetManipulator("rightarm").GetArmIndices())

def follow_trajectory_cost(target_ee_traj, old_joint_traj, robot):
    # assumes that target_traj has already been resampled
    reset_arms_to_side(robot)
    err = 0
    for lr in 'lr':
        n_steps = len(target_ee_traj[lr])
        manip_name = {"l":"leftarm", "r":"rightarm"}[lr]
        ee_link_name = "%s_gripper_tool_frame"%lr
        ee_link = robot.GetLink(ee_link_name)
        with util.suppress_stdout():
            _, pos_errs = planning.plan_follow_traj(robot, manip_name,
                                                          ee_link, 
                                                          target_ee_traj[lr],
                                                          old_joint_traj[lr])
            err += np.mean(pos_errs)            
    return err

    
