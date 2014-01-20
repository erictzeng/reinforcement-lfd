import trajoptpy, openravepy


def animate_traj(traj, robot, pause=True, step_viewer=1, restore=True, callback=None):
    """make sure to set active DOFs beforehand"""
    if restore: _saver = openravepy.RobotStateSaver(robot)
    if step_viewer or pause: viewer = trajoptpy.GetViewer(robot.GetEnv())
    for (i,dofs) in enumerate(traj):
        print "step %i/%i"%(i+1,len(traj))
        if callback is not None: callback(i)
        robot.SetActiveDOFValues(dofs)
        if pause: viewer.Idle()
        elif step_viewer:
            if i%step_viewer == 0: viewer.Step()
