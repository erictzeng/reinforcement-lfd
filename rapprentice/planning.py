import openravepy,trajoptpy, numpy as np, json
import util

def plan_follow_traj(robot, manip_name, ee_link, new_hmats, old_traj):
    orig_dof_inds = robot.GetActiveDOFIndices()
    orig_dof_vals = robot.GetDOFValues()
        
    n_steps = len(new_hmats)
    assert old_traj.shape[0] == n_steps
    assert old_traj.shape[1] == 7
    
    arm_inds  = robot.GetManipulator(manip_name).GetArmIndices()

    ee_linkname = ee_link.GetName()
    
    init_traj = old_traj.copy()

    request = {
        "basic_info" : {
            "n_steps" : n_steps,
            "manip" : manip_name,
            "start_fixed" : False
        },
        "costs" : [
        {
            "type" : "joint_vel",
            "params": {"coeffs" : [n_steps/5.]}
        },
        {
            "type" : "collision",
            "params" : {"coeffs" : [1],"dist_pen" : [0.01]}
        }                
        ],
        "constraints" : [
        ],
        "init_info" : {
            "type":"given_traj",
            "data":[x.tolist() for x in init_traj]
        }
    }

    poses = [openravepy.poseFromMatrix(hmat) for hmat in new_hmats]
    for (i_step,pose) in enumerate(poses):
        request["costs"].append(
            {"type":"pose",
             "params":{
                "xyz":pose[4:7].tolist(),
                "wxyz":pose[0:4].tolist(),
                "link":ee_linkname,
                "timestep":i_step,
                "pos_coeffs":[10,10,10],
                "rot_coeffs":[10,10,10]
             }
            })

    s = json.dumps(request)
    with openravepy.RobotStateSaver(robot):
        with util.suppress_stdout():
            prob = trajoptpy.ConstructProblem(s, robot.GetEnv()) # create object that stores optimization problem
            result = trajoptpy.OptimizeProblem(prob) # do optimization
    traj = result.GetTraj()    

    pose_costs = 0
    for (cost_type, cost_val) in result.GetCosts():
        if cost_type == 'pose':
            pose_costs += cost_val

    print "planned trajectory for %s. total pose error: %.3f."%(manip_name, pose_costs)

    # make sure this function doesn't change state of the robot
    assert not np.any(orig_dof_inds - robot.GetActiveDOFIndices())
    assert not np.any(orig_dof_vals - robot.GetDOFValues())
    
    return traj, pose_costs

