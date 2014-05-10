"""
Plotting functions using matplotlib
"""
import numpy as np
def plot_warped_grid_2d(f, mins, maxes, grid_res=None, color = 'gray', flipax = True):
    import matplotlib.pyplot as plt
    import matplotlib
    xmin, ymin = mins
    xmax, ymax = maxes
    ncoarse = 10
    nfine = 30

    if grid_res is None:
        xcoarse = np.linspace(xmin, xmax, ncoarse)
        ycoarse = np.linspace(ymin, ymax, ncoarse)
    else:
        xcoarse = np.arange(xmin, xmax, grid_res)
        ycoarse = np.arange(ymin, ymax, grid_res)
    xfine = np.linspace(xmin, xmax, nfine)
    yfine = np.linspace(ymin, ymax, nfine)

    lines = []

    sgn = -1 if flipax else 1

    for x in xcoarse:
        xy = np.zeros((nfine, 2))
        xy[:,0] = x
        xy[:,1] = yfine
        lines.append(f(xy)[:,::sgn])

    for y in ycoarse:
        xy = np.zeros((nfine, 2))
        xy[:,0] = xfine
        xy[:,1] = y
        lines.append(f(xy)[:,::sgn])        

    lc = matplotlib.collections.LineCollection(lines,colors=color,lw=2)
    ax = plt.gca()
    ax.add_collection(lc)
    plt.draw()

# almost copied from plotting_openrave
def plot_warped_grid_3d(f, mins, maxes, xres = .1, yres = .1, zres = .04, color = 'gray'):
    import matplotlib.pyplot as plt
    import matplotlib
    from mpl_toolkits.mplot3d import art3d
    xmin, ymin, zmin = mins
    xmax, ymax, zmax = maxes

    nfine = 30
    xcoarse = np.arange(xmin, xmax, xres)
    xmax = xcoarse[-1];
    ycoarse = np.arange(ymin, ymax, yres)
    ymax = ycoarse[-1];
    if zres == -1:
        zcoarse = [(zmin+zmax)/2.]
    else:
        zcoarse = np.arange(zmin, zmax, zres)
        zmax = zcoarse[-1];
    xfine = np.linspace(xmin, xmax, nfine)
    yfine = np.linspace(ymin, ymax, nfine)
    zfine = np.linspace(zmin, zmax, nfine)
    
    lines = []
    if len(zcoarse) > 1:    
        for x in xcoarse:
            for y in ycoarse:
                xyz = np.zeros((nfine, 3))
                xyz[:,0] = x
                xyz[:,1] = y
                xyz[:,2] = zfine
                lines.append(f(xyz))

    for y in ycoarse:
        for z in zcoarse:
            xyz = np.zeros((nfine, 3))
            xyz[:,0] = xfine
            xyz[:,1] = y
            xyz[:,2] = z
            lines.append(f(xyz))
        
    for z in zcoarse:
        for x in xcoarse:
            xyz = np.zeros((nfine, 3))
            xyz[:,0] = x
            xyz[:,1] = yfine
            xyz[:,2] = z
            lines.append(f(xyz))

    lc = art3d.Line3DCollection(lines,colors=color,lw=2)
    ax = plt.gca()
    ax.add_collection(lc)
    plt.draw()

def plot_correspondence(x_nd, y_nd):
    lines = np.array(zip(x_nd, y_nd))
    import matplotlib.pyplot as plt
    import matplotlib
    lc = matplotlib.collections.LineCollection(lines)
    ax = plt.gca()
    ax.add_collection(lc)
    plt.draw()
