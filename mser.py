import logging
import cv2
import numpy as np
from sets import Set
import resource
import gc

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
            root_x = self.find()
            root_y = y.find()
            if root_x == root_y:
                return root_x
            if root_x.rank < root_y.rank:
                root_x.parent = root_y
            elif root_x.rank > root_y.rank:
                root_y.parent = root_x
            else:
                root_y.parent = root_x
                root_x.rank += 1
            return root_x
        
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
        def neighbours(self, universe, image_height, image_width):
            def isClamped(p):
                if p.row >= self.row-1 and p.row <= self.row+1 and p.row >= 0 and p.row <image_height:
                    if p.col >= self.col-1 and p.col <= self.col+1 and p.col >= 0 and p.col < image_width:
                        return not(p.row == self.row and p.col == self.col)
                return False
            return filter(isClamped, universe)
    
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
            
    def __init__(self, image, threshold=32):
        self.threshold = int(threshold)
        self.image = image
        try:
            self.grey_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except:
            self.grey_scale = self.image
        
    def sorted_universe(self):
        universe = []
        logging.info("Generating universe")
        num_rows, num_cols = np.shape(self.grey_scale)
        
        for i, row in enumerate(self.grey_scale):
            usage = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
            logging.info("Working on row "+str(i)+"/"+str(num_rows)+" memory usage "+str(usage)+"MB")
            for j in xrange(len(row)):
                universe.append(MSER.Point(i,j,int(row[j])/self.threshold, row[j]))
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
        def already_processed_neighbours(point):
            alrdy_processed = filter(lambda neighbour: neighbour.intensity>=point.intensity , point.neighbours(universe,num_rows,num_cols))
            logging.info( "Point "+str(point)+"\nAlready processed "+ str(alrdy_processed))            
            return alrdy_processed
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

        logging.info("memory usage "+str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)+"MB")
        
        logging.info("Modifying tree")
        for point in universe:
            current_canonical = set1[point].find()
            current_node = set2[subtreeRoot[current_canonical.data]].find()
            logging.info("Working on "+str(current_node))
            for neighbour in already_processed_neighbours(point) :
                if neighbour.intensity >= point.intensity:
                    neighbour_canonical = set1[neighbour].find()
                    neighbour_node = set2[subtreeRoot[neighbour_canonical.data]].find()
                    logging.info("Comparing intensities of "+str(current_node.data)+ " " + str(neighbour_node.data))
                    if current_node != neighbour_node:
                        if nodes[current_node.data].level == nodes[neighbour_node.data].level :
                            # merge the nodes
                            nn = set2[neighbour_node.data] 
                            cn = set2[current_node.data]
                            temp = nn.union(cn)
                            if temp == cn :
                                nodes[cn.data].add_children(nodes[nn.data].children)
                            else:
                                nodes[nn.data].add_children(nodes[cn.data].children)
                            current_node = temp
                        else:
                            # the level is less than neighbour node's level
                            nodes[current_node.data].add_child(nodes[neighbour_node.data])
                        current_canonical = neighbour_canonical.union(current_canonical)
                        subtreeRoot[current_canonical.data] = current_node.data
            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            logging.info("memory usage "+str(usage)+"MB")
        component_map = {}
        inv_component_map = {}
        logging.info("Post processing ")
        for point in universe:
            component_map[point] = nodes[set2[point].find().data]
            
        for point in component_map.keys():
            try:
                inv_component_map[component_map[point]].append(point)
            except:
                inv_component_map[component_map[point]] = [point]
        return {"nodes":nodes[min(nodes.keys(), key=lambda node:node.intensity)], "node_map":nodes,"components":component_map, "component to points":inv_component_map}
                        
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
        