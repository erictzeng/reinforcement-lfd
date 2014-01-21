import trajoptpy, openravepy
import sys

def animate_traj(traj, robot, pause=True, step_viewer=True, restore=True, callback=None):
    """make sure to set active DOFs beforehand"""
    if restore: _saver = openravepy.RobotStateSaver(robot)
    if step_viewer or pause: viewer = trajoptpy.GetViewer(robot.GetEnv())
    for (i,dofs) in enumerate(traj):
        sys.stdout.write("step %i/%i\r"%(i+1,len(traj)))
        sys.stdout.flush()
        if callback is not None: callback(i)
        robot.SetActiveDOFValues(dofs)
        if pause: viewer.Idle()
        elif step_viewer: viewer.Step()
    sys.stdout.write("\n")
