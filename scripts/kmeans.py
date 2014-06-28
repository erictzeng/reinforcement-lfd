import numpy as np
from itertools import combinations
import cPickle as pickle
import h5py, pdb
#from do_task import *

NUM_CLUSTERS = 3
KERNEL_PARAM = 500
DS_SIZE = .025

def compute_kernel_mat(pts, func):
    kernel_mat = np.empty((len(pts), len(pts)))
    for i, a in enumerate(pts):
        for j, b in enumerate(pts[:i+1]):
            kernel_mat[i,j] = func(a, b)
            kernel_mat[j,i] = kernel_mat[i,j]
    return kernel_mat

def kmeans(pts, num_clusters, MAX_ITER=1000):
    (N, D) = pts.shape
    clusters = [list(x) for x in np.array_split(range(N), num_clusters)]
    prev = None
    ctr = 0
    while clusters != prev and ctr < MAX_ITER:
        cluster_means = [np.mean([pts[i, :] for i in c], axis = 0) for c in clusters]
        for i in range(len(cluster_means)):
            if np.any(np.isnan(cluster_means[i])):
                cluster_means[i] = np.zeros(D)
        prev = clusters
        clusters = [[] for i in range(num_clusters)]
        for point_i in range(N):
            vals = []
            for cluster_i, cluster in enumerate(prev):                
                vals.append(np.dot(pts[point_i, :], cluster_means[cluster_i]))
            clusters[np.argmin(vals)].append(point_i)
        ctr += 1
        if not ctr % 50 :
            print ctr
    return clusters, cluster_means
    

def kernelized_kmeans(kernel_mat, num_clusters):
    # initialize clusters
    N = np.shape(kernel_mat)[0]
    clusters = [list(x) for x in np.array_split(range(N), num_clusters)]
    prev = None
    while clusters != prev:
        # precompute pairwise dot products
        cluster_weights = []
        cluster_widths = []
        for cluster in clusters:
            cluster_weights.append(sum(kernel_mat[b, c] for b, c in combinations(cluster, 2)) \
                / len(cluster) ** 2)
            cluster_widths.append(0)
        prev = clusters
        clusters = [[] for i in range(num_clusters)]
        for point_i in range(N):
            vals = []
            for cluster_i, cluster in enumerate(prev):
                val = cluster_weights[cluster_i] + kernel_mat[point_i, point_i]
                val -= 2 * sum(kernel_mat[point_i, b] for b in cluster) / len(cluster)
                vals.append(val)
            assigned_cluster = np.argmin(vals)
            clusters[assigned_cluster].append(point_i)
            if min(vals) > cluster_widths[assigned_cluster]: 
                cluster_widths[assigned_cluster] = min(vals)
    return clusters, cluster_widths

def build_reverse_ind(keys, clusters):
    reverse_ind = {}
    for i, cluster in enumerate(clusters):
        for pt_i in cluster:
            reverse_ind[keys[pt_i]] = i
    return reverse_ind
 
def rbf(arr, sigma_sq=250):
    return np.exp(-np.square(arr) / (2 * sigma_sq))

def load_clusters(fname):
    with open(fname, 'rb') as infile:
        obj = pickle.load(infile)
    return obj

def compute_distance_mat():
    demofile = h5py.File('/home/dhm/sampledata/overhand/all.h5', 'r')
    ds_clouds = [clouds.downsample(seg["cloud_xyz"], DS_SIZE) for seg in demofile.values()]

    func = lambda a, b: registration_cost(a, b) + registration_cost(b, a)
    kernel_mat = compute_kernel_mat(ds_clouds, func)
    outfile = h5py.File('kernel_mat.h5', 'w')
    outfile['mat'] = kernel_mat
    outfile.close()
