#!/usr/bin/env python

# Generates synthetic warps of images, by the following method:
#     1) Create point cloud by randomly sampling points from the original image
#        and only keeping the **non-white** points
#     2) Warp the original point cloud by each of the prespecified warps, to get
#        new warped point clouds (associated with the same RGBA data).
#
# Prints out data in the specified output h5 file, with the following keys:
#     "orig_cloud" -> numpy array with columns x-coord, y-coord, r, g, b, a
#     "warp_cloud_<ID>" -> numpy array with same format as above

import argparse, h5py, Image, math, numpy as np
import scipy.spatial.distance as ssd

WHITE = (255, 255, 255)
Z_MULT = 30.0  # range of z = min(x_axis_pixels, y_axis_pixels) / Z_MULT
SCALE = 500.0
RGB_SCALE = 255.0
THETA_VALS = []
S_VALS = []
#S_VALS = [0.01, 0.02, 0.03, 0.04, 0.05]
ALPHA = 0.5

def get_full_point_cloud(pixels, x, y, add_z):
    # Returns numpy array of the point cloud (x, y, r, g, b) with all non-white
    # points in the image
    z_scale = min(x,y) / Z_MULT
    points = []
    for i in xrange(x):
        for j in xrange(y):
          new_pt_info = pixels[i, j][:3]
          if new_pt_info == WHITE:
              continue
          if add_z:
              new_pt = (i,j) + (np.random.rand()*z_scale,)
          new_pt = tuple(coord / SCALE for coord in new_pt)
          new_pt_info = tuple(val / RGB_SCALE for val in new_pt_info)
          points.append(new_pt + new_pt_info)
    return np.asarray(points)

def get_stratified_sampled_point_cloud(pixels, x, y, add_z, stratified_size, iter_limit = 10000):
    # Randomly samples, but not more than stratified_size points of each color
    z_scale = min(x,y) / Z_MULT
    point_set = set()
    pts_per_color = {}
    for i in xrange(iter_limit):
        new_pt = (np.random.randint(x), np.random.randint(y))
        new_pt_info = pixels[new_pt[0], new_pt[1]][:3]
        if new_pt_info == WHITE:
            continue
        if add_z:
            new_pt = new_pt + (np.random.rand()*z_scale,)
        if new_pt_info in pts_per_color:
            pts_per_color[new_pt_info] += 1
            if pts_per_color[new_pt_info] >= stratified_size:
                continue
        else:
            pts_per_color[new_pt_info] = 1
        new_pt = tuple(coord / SCALE for coord in new_pt)
        new_pt_info = tuple(val / RGB_SCALE for val in new_pt_info)
        point_set.add(new_pt + new_pt_info)
    return np.asarray(list(point_set))

def get_sampled_point_cloud(pixels, x, y, add_z, target_size):
    # Returns numpy array of the point cloud (x, y, r, g, b) created by
    # randomly sampling non-white points from the image
    z_scale = min(x,y) / Z_MULT
    point_set = set()
    while len(point_set) < target_size:
        new_pt = (np.random.randint(x), np.random.randint(y))
        new_pt_info = pixels[new_pt[0], new_pt[1]][:3]  # Ignore alpha
        if new_pt_info != WHITE:
            if add_z:
                new_pt = new_pt + (np.random.rand()*z_scale,)
            new_pt = tuple(coord / SCALE for coord in new_pt)
            new_pt_info = tuple(val / RGB_SCALE for val in new_pt_info)
            point_set.add(new_pt + new_pt_info)
    return np.asarray(list(point_set))

def get_point_cloud(imagefile, add_z, target_size = None, stratified_size = None):
    im = Image.open(imagefile)
    pixels = im.load()
    x,y = im.size
    if target_size is None and stratified_size is None:
        return get_full_point_cloud(pixels, x, y, add_z)
    if target_size is None and stratified_size is not None:
        return get_stratified_sampled_point_cloud(pixels, x, y, add_z, stratified_size)
    else:
        return get_sampled_point_cloud(pixels, x, y, add_z, target_size)

def randomly_warp_point_cloud(cloud, s):
    # The weights of the Gaussian RBF are randomly drawn from N(0, s^2)
    # cloud: Numpy array with XYZRGB info per row (or XYZRGB)
    # resulting warp: f(x) = x + \sum_{i=1}^N \lambda_i \phi(||x-x_i||_2)
    n,d = cloud.shape
    d -= 3  # To account for the three RGB coordinates
    cloud_xy = cloud[:,:2]

    weights_x = np.random.normal(0,s,n)
    weights_y = np.random.normal(0,s,n)
    pdists = np.exp(-1*ALPHA*ssd.squareform(ssd.pdist(cloud_xy, 'sqeuclidean')))
    warped_cloud_xy = cloud.copy()
    warped_cloud_xy[:,:2] = cloud_xy + np.transpose(np.vstack((weights_x.dot(pdists), weights_y.dot(pdists))))
    return warped_cloud_xy

def rotate_point_cloud(cloud, theta):
    # theta must be in degrees
    theta = math.radians(theta)
    rot_matrix = np.asarray([[math.cos(theta), -1*math.sin(theta)], \
                             [math.sin(theta), math.cos(theta)]])
    centroid = np.mean(cloud[:,:2],axis = 0)
    rotated_cloud = cloud.copy()
    rotated_cloud[:,:2] = (cloud[:,:2] - centroid).dot(rot_matrix) + centroid
    return rotated_cloud

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_image", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("--test_image", type=str)
    parser.add_argument("--target_size", type=int, nargs='?', default=100)
    parser.add_argument("--stratified_size", type=int)
    parser.add_argument("--add_z",  action="store_true")

    args = parser.parse_args()

    dem_cloud = get_point_cloud(args.input_image, args.add_z, stratified_size = args.stratified_size)

    output = h5py.File(args.output_file, 'w')
    output['dem_cloud'] = dem_cloud

    if args.test_image:
        test_cloud = get_point_cloud(args.test_image, args.add_z, stratified_size = args.stratified_size)
        output['warp_test_cloud'] = test_cloud
        output['warp_rot180_test_cloud'] = rotate_point_cloud(test_cloud, 180)

    for t in THETA_VALS:
        key = 'warp_rot' + str(t)
        output[key] = rotate_point_cloud(dem_cloud, t)

    rot45_dem_cloud = rotate_point_cloud(dem_cloud, 45)
    for s in S_VALS:
        key = 'warp_s' + str(s).replace('.','_')
        output[key] = randomly_warp_point_cloud(dem_cloud, s)
        key = 'rot45_warp_s' + str(s).replace('.','_')
        output[key] = randomly_warp_point_cloud(rot45_dem_cloud, s)
    output.close()

if __name__ == "__main__":
    main()
