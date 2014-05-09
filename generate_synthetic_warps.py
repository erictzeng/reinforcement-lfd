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

WHITE = (255, 255, 255)

def get_point_cloud(imagefile, target_size):
    # Returns numpy array of the point cloud (x, y, r, g, b, a) created by
    # randomly sampling non-white points from the file
    im = Image.open(imagefile)
    pixels = im.load()
    x,y = im.size
    point_set = set()
    while len(point_set) < target_size:
        new_pt = (np.random.randint(x), np.random.randint(y))
        new_pt_info = pixels[new_pt[0], new_pt[1]][:3]  # Ignore alpha
        if new_pt_info != WHITE:
            point_set.add(new_pt + new_pt_info)

    return np.asarray(list(point_set))

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

    args = parser.parse_args()

    orig_cloud = get_point_cloud(args.input_image, args.target_size)
    warp_rot180 = rotate_point_cloud(orig_cloud, math.pi)
    # TODO: Generate random rotations (see how Chui et al. did in TPS-RPM paper)

    output = h5py.File(args.output_file, 'w')
    output['orig_cloud'] = orig_cloud
    output['warp_cloud_rot180'] = warp_rot180
    output.close()

if __name__ == "__main__":
    main()
