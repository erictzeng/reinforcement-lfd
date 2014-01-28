#!/usr/bin/env python

import numpy as np

#
# line segment intersection using vectors
# see Computer Graphics by F.S. Hill
#
def perp( a ) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
# return 
def seg_intersect(p1,p2,p3,p4) :
    p1=np.float32(p1)
    p2=np.float32(p2)
    p3=np.float32(p3)
    p4=np.float32(p4)
    numa = (p4-p3).dot(perp(p3-p1))
    numb = (p2-p1).dot(perp(p3-p1))
    denom = (p2-p1).dot(perp(p3-p4))
    if denom == 0:
        if numa==0 or numb==0: # coincident lines
            return (0.5,0.5)
        else: # parallel lines
            return None
    ua = (numa / denom)
    ub = (numb / denom)
    if ua >= 0 and ua <= 1 and ub >= 0 and ub <= 1:
        return (ua,ub)
    else:
        return None

def calculateCrossings(rope_nodes):
    crossings = np.zeros(rope_nodes.shape[0]-1) # 1 for overcrossings and -1 for undercrossings
    for i_node in range(rope_nodes.shape[0]-1):
        for j_node in range(i_node+2,rope_nodes.shape[0]-1):
            intersect = seg_intersect(rope_nodes[i_node,:2], rope_nodes[i_node+1,:2], rope_nodes[j_node,:2], rope_nodes[j_node+1,:2])
            if intersect:
                i_link_z = rope_nodes[i_node,2] + intersect[0] * (rope_nodes[i_node+1,2] - rope_nodes[i_node,2])
                j_link_z = rope_nodes[j_node,2] + intersect[1] * (rope_nodes[j_node+1,2] - rope_nodes[j_node,2])
                i_over_j = i_link_z > j_link_z
                crossings[i_node] = 1 if i_over_j else -1
                crossings[j_node] = 1 if not i_over_j else -1
    return crossings

def crossingsToString(crossings):
    s = ''
    for c in crossings:
        if c == 1:
            s += 'o'
        elif c == -1:
            s += 'u'
    return s

def isKnot(rope_nodes):
    crossings = calculateCrossings(rope_nodes)
    s = crossingsToString(crossings)
    knot_topologies = ['uououo', 'uoouuoou']
    for top in knot_topologies:
        if top in s:
            return True
        if top[::-1] in s:
            return True
        flipped_top = top.replace('u','t').replace('o','u').replace('t','o')
        if flipped_top in s:
            return True
        if flipped_top[::-1] in s:
            return True
    return False