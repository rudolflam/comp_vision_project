# -*- coding: utf-8 -*-
"""
@author: rudolf
"""

# Alternate implementation of mser from another paper

import platform
linux = False
if platform.system() == "Linux":
    linux = True
    import resource
import logging
import cv2
import numpy as np
class Point(object):
        __slots__ = ["i","row", "col", "intensity", "pixel_intensity"]
        def __init__(self, i, row, col, intensity, pixel_intensity):
            self.i = i
            self.row = row
            self.col = col
            self.intensity = intensity
            self.pixel_intensity = pixel_intensity
        def __str__(self):
            return "("+str(self.row) + "," + str(self.col)+")@intensity:"+str(self.pixel_intensity)+" "
        def __repr__(self):
            return str(self)
        def index(self):
            return self.row * self.cols + self.col
        def neighbours(self,higher_intensity) :
            return filter(lambda n: abs(n.row-self.row) + abs(n.col-self.col) == 1 and (n.row!=self.row and n.col!=self.col),higher_intensity)
class ComponentTree(object):
        __slots__=["children", "level","area", "highest", "rank", "parent"]
        def __init__(self, level):
            self.children = []
            self.level = level
            self.area = 1
            self.highest= level
        def __str__(self):
            return "("+str(self.level) + ":" + str(self.children)+") "
        def __repr__(self):
            return str(self)
            
        def add_child(self, child):
            self.children.append(child)
            
        def add_children(self, children):
            for child in children:
                self.children.append(child)  
                
def get_resource():
    if linux:
        return str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    return "None available"
    
def sorted_universe(image,threshold):
        universe = []
        logging.info("Generating universe")
        grey_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        num_rows, num_cols = np.shape(grey_scale)
        
        for i, row in enumerate(grey_scale):
            logging.info("Working on row "+str(i)+"/"+str(num_rows)+" memory usage "+get_resource()+"MB")
            for j in xrange(len(row)):
                universe.append(Point(i*num_cols+j,i,j,int(row[j])/threshold, row[j]))
        logging.info("Sorting universe")
        return sorted(universe, key=lambda point:point.intensity, reverse=True)


def MakeSet_node(x,nodes_parent, nodes_rank):
    nodes_parent[x] = x
    nodes_rank[x] = 0
def MakeSet_tree(x, tree_parent, tree_rank):
    tree_parent[x] = x
    tree_rank[x] = 0

def Find_node(x, nodes_parent):
    if nodes_parent[x] != x:
        nodes_parent[x] = Find_node(nodes_parent[x], nodes_parent)
    return nodes_parent[x]
def Find_tree(x, tree_parent):
    if tree_parent[x] != x:
        tree_parent[x] = Find_tree(tree_parent[x], tree_parent)
    return tree_parent[x]

def Link_node(x,y,nodes,nodes_rank):
    xref = nodes[x]
    yref = nodes[y]
    xref_rank = nodes_rank[x]
    yref_rank = nodes_rank[y]
    if xref_rank > yref_rank:
        temp = xref
        xref = yref
        yref = temp
        t = x
        x = y 
        y = t
    if xref_rank  == yref_rank:
        yref_rank = yref_rank+1
    xref.parent = yref
    return y
def Link_tree(x,y,tree, tree_rank):
    xref = tree[x]
    yref = tree[y]
    xref_rank = tree_rank[x]
    yref_rank = tree_rank[y]
    if xref_rank > yref_rank:
        temp = xref
        xref = yref
        yref = temp
        t = x
        x = y 
        y = t
    if xref_rank  == yref_rank:
        yref_rank = yref_rank+1
    xref.parent = yref
    return y
def MakeNode(level):
    return ComponentTree(level)
    
def MergeNodes(node1, node2, nodes, nodes_rank):
    tmpNode = Link_node(node1,node2, nodes, nodes_rank)
    tmpNode2 = None
    if tmpNode == nodes[node2]:
        nodes[node2].add_children(nodes[node1].children)
        tmpNode2 = node1
    else :
        nodes[node1].add_children(nodes[node2].children)
        tmpNode2 = node2
    nodes[tmpNode].area += nodes[tmpNode2].area
    nodes[tmpNode].highest = max([nodes[tmpNode].highest, nodes[tmpNode2].highest])    
    return tmpNode

def already_processed_neighbours(point,higher_intensity,points):
    if point.intensity < higher_intensity[-1].intensity:
        higher_intensity = filter(lambda neighbour: neighbour.intensity>=point.intensity , points)        
    return point.neighbours(higher_intensity) 

def build_component_tree(points,image):
    num_points = len(points)
    tree_parent = [None]*num_points
    tree_rank = [None]*num_points
    node_parent = [None]*num_points
    node_rank = [None]*num_points
    nodes = [None]*num_points
    tree = [None]*num_points
    lowest_node = [None]*num_points
    higher_intensity = filter(lambda neighbour: neighbour.intensity>=points[0].intensity , points)
    for i,p in enumerate(points) :
        MakeSet_tree(i, tree_parent, tree_rank)
        MakeSet_node(i, node_parent, node_rank)
        nodes[i] = MakeNode(p.intensity)
        tree[i] = MakeNode(p.intensity)
        lowest_node[i] = i
    for i,p in enumerate(points):
        curTree = Find_tree(i, tree_parent)
        curNode = Find_node(lowest_node[curTree],node_parent)
        for q in already_processed_neighbours(p, higher_intensity, points):
            adjTree = Find_tree(q.i, tree_parent)
            adjNode = Find_node(lowest_node[adjTree], node_parent)
            if curNode != adjNode:
                if nodes[curNode].level == nodes[adjNode].level:
                    curNode = MergeNodes(adjNode, curNode, nodes, node_rank)
                else:
                    nodes[curNode].add_child(nodes[adjNode])
                    nodes[curNode].area += nodes[adjNode].area
                    nodes[curNode].highest = max(nodes[curNode].highest, nodes[adjNode].highest)
                curTree = Link_tree(adjTree, curTree, tree, tree_rank)
                lowest_node[curTree] = curNode
    print node_parent, " " , tree_parent
    root = lowest_node[Find_tree(Find_node(0, node_parent),tree_parent)]
    component_map = [None]*num_points
    for i,p in enumerate(points):
        component_map[i] = Find_node(i, node_parent)
    return {"nodes":nodes, "root":root, "component_map":component_map}