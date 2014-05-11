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

def get_point_cloud(imagefile, target_size, add_z):
    # Returns numpy array of the point cloud (x, y, r, g, b, a) created by
    # randomly sampling non-white points from the file
    im = Image.open(imagefile)
    pixels = im.load()
    x,y = im.size
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

def randomly_warp_point_cloud(cloud, s):
    # The weights of the Gaussian RBF are randomly drawn from N(0, s^2)
    # cloud: Numpy array with XYZRGB info per row (or XYZRGB)
    # resulting warp: f(x) = x + \sum_{i=1}^N \lambda_i \phi(||x-x_i||_2)
    n,d = cloud.shape
    d -= 3  # To account for the three RGB coordinates
    cloud_xy = cloud[:,:2]

    weights_x = np.random.normal(0,s,n)
    weights_y = np.random.normal(0,s,n)
    pdists = ssd.squareform(ssd.pdist(cloud_xy, 'euclidean'))
    warped_cloud_xy = cloud.copy()
    #import IPython as ipy
    #ipy.embed()
    warped_cloud_xy[:,:2] = cloud_xy + np.transpose(np.vstack((weights_x.dot(pdists), weights_y.dot(pdists))))
    return warped_cloud_xy

def rotate_point_cloud(cloud, theta):
    # theta must be in radians
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
    parser.add_argument("--target_size", type=int, nargs='?', default=100)
    parser.add_argument("--add_z",  action="store_true")

    args = parser.parse_args()

    orig_cloud = get_point_cloud(args.input_image, args.target_size, args.add_z)
    dem_cloud = get_point_cloud(args.input_image, args.target_size*2, args.add_z)

    warp_rot30 = rotate_point_cloud(dem_cloud, math.pi/6)
    warp_rot60 = rotate_point_cloud(dem_cloud, math.pi/3)
    warp_rot180 = rotate_point_cloud(dem_cloud, math.pi)

    warp_s0_001 = randomly_warp_point_cloud(dem_cloud, 0.001)
    warp_s0_01 = randomly_warp_point_cloud(dem_cloud, 0.01)
    warp_s0_1 = randomly_warp_point_cloud(dem_cloud, 0.1)

    output = h5py.File(args.output_file, 'w')
    output['orig_cloud'] = orig_cloud
    output['dem_cloud'] = dem_cloud
    output['warp_cloud_rot30'] = warp_rot30
    output['warp_cloud_rot60'] = warp_rot60
    output['warp_cloud_rot180'] = warp_rot180
    output['warp_s0_001'] = warp_s0_001
    output['warp_s0_01'] = warp_s0_01
    output['warp_s0_1'] = warp_s0_1
    output.close()

if __name__ == "__main__":
    main()
