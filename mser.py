import logging
import cv2
import numpy as np
from sets import Set
import platform
linux = False
if platform.system() == "Linux":
    linux = True
    import resource
import gc

def get_resource():
    if linux:
        return str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    return "None available"

class MSER:
    @staticmethod
    def logging_on():
        logging.getLogger().setLevel(logging.getLevelName("INFO"))
    
    @staticmethod
    def load_example():
        return np.array([[110,90,100],[50]*3, [40,20,50], [50]*3, [120,70,80]])
    
    class DisjointSet(object):
        __slots__ = ["parent","rank","data"]
        def __init__(self, data):
            self.parent = self
            self.rank = 0
            self.data = data
        
        def find(self):
            if self.parent != self:
                self.parent = self.parent.find()
            return self.parent
        
        def union(self, y):
            root_x = self
            root_y = y
            if root_x.rank > root_y.rank:
                temp = root_x
                root_x = root_y
                root_y = temp
            if root_x.rank == root_y.rank:
                root_y.rank += 1
            root_x.parent = root_y
            return root_y
#            root_x = self.find()
#            root_y = y.find()
#            if root_x == root_y:
#                return root_x
#            if root_x.rank < root_y.rank:
#                root_x.parent = root_y
#            elif root_x.rank > root_y.rank:
#                root_y.parent = root_x
#            else:
#                root_y.parent = root_x
#                root_x.rank += 1
#            return root_x
        
    class Point(object):
        __slots__ = ["row", "col", "intensity", "pixel_intensity"]
        def __init__(self, row, col, intensity, pixel_intensity):
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
        def neighbours(self, universe):
#            def isClamped(p):
#                if p.row >= self.row-1 and p.row <= self.row+1:
#                    if p.col >= self.col-1 and p.col <= self.col+1:
#                        return not(p.row == self.row and p.col == self.col)
#                return False
#                if p.row >= self.row-1 and p.row <= self.row+1 and p.row >= 0 and p.row <image_height:
#                    if p.col >= self.col-1 and p.col <= self.col+1 and p.col >= 0 and p.col < image_width:
#                        return not(p.row == self.row and p.col == self.col)
#                return False
#            return filter(isClamped, universe)
            return filter(lambda n: abs(n.row-self.row)+abs(n.col-self.col)==1,universe)
    
    class ComponentTree(object):
        __slots__=["children", "level"]
        def __init__(self, level):
            self.children = []
            self.level = level
        def __str__(self):
            return "("+str(self.level) + ":" + str(self.children)+") "
        def __repr__(self):
            return str(self)
            
        def add_child(self, child):
            self.children.append(child)
            
        def add_children(self, children):
            for child in children:
                self.children.append(child)
            
    def __init__(self, image, threshold=32,inv=True):
        self.threshold = int(threshold)
        self.image = image
        try:
            self.grey_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if inv:
                for i,row in enumerate(self.grey_scale):
                    for j,e in enumerate(row):
                        self.grey_scale[i,j] = 255 - e
        except:
            self.grey_scale = self.image
        
    def sorted_universe(self):
        universe = []
        logging.info("Generating universe")
        num_rows, num_cols = np.shape(self.grey_scale)
        
        for i, row in enumerate(self.grey_scale):
            logging.info("Working on row "+str(i)+"/"+str(num_rows)+" memory usage "+get_resource()+"MB")
            for j in xrange(len(row)):
                universe.append(MSER.Point(i,j,int(row[j])/self.threshold+1, row[j]))
#            gc.collect()
#            for j,element in enumerate(row):
#                universe.append(MSER.Point(i,num_rows,j, num_cols,int(element)/self.threshold, element))
        logging.info("Sorting universe")
        return sorted(universe, key=lambda point:point.intensity, reverse=True)
        
    def build_tree(self, universe):
        logging.info("Building tree")
        nodes = {}
        subtreeRoot = {}
        set1 = {}
        set2 = {}
        num_rows, num_cols = np.shape(self.grey_scale)
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
            logging.info("Working on "+str(i)+"/"+str(num_points)+"("+str(i*100/num_points)+"%) point")

        logging.info("memory usage "+get_resource()+"MB")
        
        logging.info("Modifying tree")
        i = 0
        for point in universe:
            i += 1
            current_canonical = set1[point].find()
            current_node = set2[subtreeRoot[current_canonical.data]].find()
            logging.info("Modifying tree for point "+str(i)+"/"+str(num_points)+"("+str(i*100/num_points)+"%) point ")
            print point, " ", point.intensity
            for neighbour in already_processed_neighbours(point,higher_intensity) :
                if neighbour.intensity >= point.intensity:
                    neighbour_canonical = set1[neighbour].find()
                    neighbour_node = set2[subtreeRoot[neighbour_canonical.data]].find()
#                    logging.info("Comparing intensities of "+str(current_node.data)+ " " + str(neighbour_node.data))
                    if current_node != neighbour_node:
                        if nodes[current_node.data].level == nodes[neighbour_node.data].level :
                            # merge the nodes
                            nn = set2[neighbour_node.data] 
                            cn = set2[current_node.data]
                            temp = nn.union(cn)
                            if temp == cn :
                                nodes[cn.data].add_children(nodes[nn.data].children)
                                nodes[nn.data].children = nodes[cn.data].children
                            else:
                                nodes[nn.data].add_children(nodes[cn.data].children)
                                nodes[cn.data].children = nodes[nn.data].children
                            current_node = temp
                        else:
                            # the level is less than neighbour node's level
                            nodes[current_node.data].add_child(nodes[neighbour_node.data])
                        current_canonical = neighbour_canonical.union(current_canonical)
                        subtreeRoot[current_canonical.data] = current_node.data
            logging.info("memory usage "+get_resource()+"MB")
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
#            try:
#                inv_component_map[component_map[point]].append(point)
#            except:
#                inv_component_map[component_map[point]] = [point]
        
        root_point = subtreeRoot[set1[set2[universe[0]].find().data].find().data]
        print "Root point ", root_point, " ", root_point.intensity
        total_len = 0        
        for c in inv_component_map.keys():
            total_len += len(inv_component_map[c])
        print "Total points in inv ",total_len
        get_points_from_subtree(nodes[root_point], inv_component_map)
        return {"nodes":nodes[root_point], "node_map":nodes,"components":component_map, "component to points":inv_component_map}
                        
    def build_component_tree(self):
        # add all points in image to a list
        logging.info("Setting up component tree data structures")
        universe = self.sorted_universe()
        logging.info("Building tree")
        tree = self.build_tree(universe)
        return {"points":universe, "root":tree["nodes"], "nodes":tree["node_map"], "components":tree["components"], "component to points":tree["component to points"]}
    
    @staticmethod
    def extremal_region(component_to_points, component):
        points = Set()
        for point in component_to_points[component]:
            points.add(point)
        if component.children:
            for child in component.children:
                for point in MSER.extremal_region(component_to_points, child):
                    points.add(point)
        return points
        