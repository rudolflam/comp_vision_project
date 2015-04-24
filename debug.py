# -*- coding: utf-8 -*-
"""
@author: rudolf
"""

import mser
import cv2
import numpy as np
import logging
from mser import MSER
from sets import Set
cv2.namedWindow("image", cv2.WINDOW_NORMAL)
image = cv2.imread("vsmall.jpg")
cv2.imshow("image" , image)
m = mser.MSER(image)
universe = m.sorted_universe()
nodes = {}
subtreeRoot = {}
set1 = {}
set2 = {}
num_rows, num_cols = np.shape(m.grey_scale)
higher_intensity = filter(lambda neighbour: neighbour.intensity>=universe[0].intensity , universe)
def already_processed_neighbours(point,higher_intensity):
    if point.intensity < higher_intensity[-1].intensity:
        higher_intensity = filter(lambda neighbour: neighbour.intensity>=point.intensity , universe)        
    return point.neighbours(higher_intensity) 
logging.info("Preprocessing step")
num_points = len(universe)
i = 0
for point in universe:
    i += 1
    set1[point] = MSER.DisjointSet(point)
    set2[point] = MSER.DisjointSet(point)
    nodes[point] = MSER.ComponentTree(point.intensity)
    subtreeRoot[point] = point
i = 0
print "Universe bomb"
for point in universe:
    i += 1
    current_canonical = set1[point].find()
    current_node = set2[subtreeRoot[current_canonical.data]].find()
#    print "Looking at ", nodes[current_node.data]
#    if point.row == 22 and point.col == 19:
#                    print "Looking at special point"
    for neighbour in already_processed_neighbours(point,higher_intensity) :
#        if point.row == 22 and point.col == 19:
#            print "Neighbour of special pt ", neighbour, " ", neighbour.intensity
        if neighbour.intensity >= point.intensity:
            neighbour_canonical = set1[neighbour].find()
            print "node : ", point, " neighbour : ", neighbour
            if point.row == 0 and point.col == 7:
                print ">>>>current canonical ", current_canonical.data
                print ">>>>Neighbour canonical ", neighbour_canonical.data
            else : 
                print "current canonical ", current_canonical.data
                print "neighbour canonical ", neighbour_canonical.data
            neighbour_node = set2[subtreeRoot[neighbour_canonical.data]].find()
#                    logging.info("Comparing intensities of "+str(current_node.data)+ " " + str(neighbour_node.data))
            if current_node != neighbour_node:
                if nodes[current_node.data].level == nodes[neighbour_node.data].level :
                    if point.row == 0 and point.col == 7:
                        print ">>>>merging ", current_node.data, " & ",neighbour_node.data , " @ ",nodes[current_node.data].level
                    else:
                        print "merging ",current_node.data, " & ",neighbour_node.data , " @ ",nodes[current_node.data].level
                    # merge the nodes
                    nn = set2[neighbour_node.data] 
                    cn = set2[current_node.data]
                    temp = nn.union(cn)
                    if temp is cn :
                        nodes[cn.data].add_children(nodes[nn.data].children)
                        nodes[nn.data].children = nodes[cn.data].children
                    else:
                        nodes[nn.data].add_children(nodes[cn.data].children)
                        nodes[cn.data].children = nodes[nn.data].children
                    current_node = temp
#                    print "Merging neigbhour ", neighbour, " and point ", point
#                    print "To ",nodes[current_node.data]
                else:
                    # the level is less than neighbour node's level
                    nodes[current_node.data].add_child(nodes[neighbour_node.data])  
                current_canonical = neighbour_canonical.union(current_canonical)
                subtreeRoot[current_canonical.data] = current_node.data
                print "Subtree root ", current_node.data," given to ", subtreeRoot[current_canonical.data]
#                print "Subtree roots ", subtreeRoot
                print "==============="
                print "Subtree built ", nodes[point]
                print "==============="
print subtreeRoot
component_map = {}
inv_component_map = {}
logging.info("Post processing ")
for point in universe:
    component_map[point] = nodes[set2[point].find().data]

def give_point_to_component(point,component,inv_component_map):
    try:
        inv_component_map[component_map[point]].add(point)
    except:
        inv_component_map[component_map[point]] = Set([point])
    for child in component.children:
        give_point_to_component(point, child,inv_component_map)
    
def get_points_from_subtree(tree,inv_component_map):
    for c in tree.children:
        get_points_from_subtree(c, inv_component_map)
    for c in tree.children:
        inv_component_map[tree] |= inv_component_map[c]

for point in component_map.keys():
    give_point_to_component(point, component_map[point],inv_component_map)
    
root_point = subtreeRoot[set1[set2[universe[0]].find().data].find().data]
root_node = nodes[root_point]

copy = image.copy()
from main import TextDetector as td
td.draw_points(image, inv_component_map[root_node])