#!/usr/bin/python
import logging
import cv2
import numpy as np
import argparse
import glob
import os
import copy
import string
from mser import MSER

def show_image(image, window_name="Image"):
    image = cv2.resize(image, (0,0), fx=0.5,fy=0.5)
    cv2.imshow(window_name, image)
    while( cv2.waitKey() != ord("q")):
        pass
    cv2.destroyAllWindows()

class TextDetector(object):
    @staticmethod    
    def run_MSER(image, show=False):
        mser = MSER(image, 10)
        data = mser.build_component_tree()
        universe = data["points"]
        root_node = data["root"]
        nodes= data["nodes"]
        print data.keys()
        components_to_points = data["component to points"]
        
        # prune the excess regioins
        mser_regions = TextDetector.prune_MSERs(mser.grey_scale, universe, root_node, nodes, components_to_points)

#        if show:
#            hulls = [cv2.convexHull(p.reshape(-1,1,2)) for p in mser_regions ]
#             # display the image
#            display = image.copy()
#            cv2.polylines(display, hulls, 1, (0,255,0))
#            show_image(display)
        return mser_regions
    
    @staticmethod
    def prune_MSERs(image, points, root_node, nodes, component_to_points):
        
        
        a_max = 1.2
        a_min = 0.7
        theta1= 0.03
        theta2= 0.08        
        
        class ER:
            def __init__(self, component):
                self.component = component
                self.variation = 1.0
                self.children = []
            def add_child(self, child):
                self.children.append(child)
            def __str__(self):
                return "("+str(self.variation) + ":" + str(self.children)+") "
            def __repr__(self):
                return str(self)
        def difference(component1, component2):
            return MSER.extremal_region(component_to_points, component1).difference(MSER.extremal_region(component_to_points, component2))
        def size(component):
            return len(MSER.extremal_region(component_to_points, component))
        def variation(componentDelta, component):
            component_size=size(component)
            return abs(size(componentDelta)-component_size) / component_size
        def aspect_ratio(component):
            copy = image.copy()
            points_in_region = MSER.extremal_region(component_to_points, component)
            num_rows, num_cols = np.shape(copy)
            min_row = num_rows
            max_row = 0
            min_col = num_cols
            max_col = 0
            for point in points_in_region:
                if point.row < min_row:
                    min_row = point.row
                elif point.row > max_row:
                    max_row = point.row
                if point.col < min_col:
                    min_col = point.col
                elif point.col > max_col:
                    max_col = point.col
            width, height = (max_row-min_row) , (max_col-min_col)
            if width > 0 and height>0:
                return float(width)/height
            return 10000

        def to_ER_tree(parent_component, parent_ER):
            # performs regularization simutaneously
            for child in parent_component.children:
                child_ER = ER(child)
                v = variation(parent_component, child)
                a = aspect_ratio(child)
                if a:
                    child_ER.variation = v - theta1*(a-a_max) if a > a_max else v - theta2*(a_min-a) if a<a_min else v

                else:
                    child_ER.variation = 100000
                parent_ER.add_child(child_ER)
                if child.children:
                    to_ER_tree(child, child_ER)
        def linear_reduction(tree):
            if len(tree.children)==0:
                return tree
            elif len(tree.children)==1:
                c = linear_reduction(tree.children[0])
                if tree.variation<= c.variation:
                    tree.children = c.children
                    return tree
                else:
                    return c
            else:
                for index in range(len(tree.children)):
                    tree.children[index] = linear_reduction(tree.children[index])
                return tree
        def tree_acum(tree):
            if len(tree.children) >= 2:
                C = []
                min_var = tree_acum(tree.children[0]).variation;
                for index in range(len(tree.children)):
                    C = C+tree_acum(tree.children[index])
                    if tree_acum(tree.children[index]).variation<min_var:
                        min_var =  tree_acum(tree.children[index]).variation
                if tree.variation <= min_var:
                    tree.children = []
                    return tree
                else:
                    return C
            else:
                return tree
        def tree_accumulation(tree):
            if len(tree.children) >= 2:
                C = []
                for c in tree.children:
                    C = C + tree_accumulation(c)
                if tree.variation <= min(C, key=lambda c: c.variation):
                    tree.children = []
                    return [tree]
                else:
                    return C
            else:
                return [tree]
        current_component = root_node
        parent_ER = ER(current_component)
        to_ER_tree(root_node, parent_ER)
        print "ER Tree ", parent_ER
        lr = linear_reduction(parent_ER)
        print len(lr.children)
        print "ER Tree after linear reduction ", lr
        acc = tree_accumulation(lr)
        print "ER Tree after accumulation ", acc
        return acc
        
    @staticmethod 
    def candidates_selection(mser_regions):
        #TODO
        candidates = []
        return candidates
        
    @staticmethod 
    def good_candidate(candidate):
        #TODO
        return True
        
    @staticmethod
    def candidates_elimination(candidates):
        candidates = filter(lambda candidate: TextDetector.good_candidate(candidate), candidates)
        return candidates

class Dataset(object):
    data_set_path = "./MSRA-TD500/"
    training_set_path = "./MSRA-TD500/train/"
    test_set_path = "./MSRA-TD500/test/"
    jpg_ext = ".JPG"
    TRAINING = 1
    TEST = 0
    
    @staticmethod
    def read_image(filename, which_set=1, show=False):
        image_path = Dataset.training_set_path + filename if which_set else Dataset.test_set_path+filename
        image = cv2.imread(image_path)
        if show:
            show_image(image)
        return image
        
    @staticmethod
    def rotated_rect(x,y,w,h,angle):
        s,c = np.sin(angle), np.cos(angle)
        mean_x,mean_y = x + w / 2, y + h / 2
        w_2,h_2 = w/2,h/2
        top_left = Dataset.rotate_and_offset(-w_2,-h_2,s,c,mean_x,mean_y)
        top_right = Dataset.rotate_and_offset(w_2,-h_2,s,c,mean_x,mean_y)
        bottom_right = Dataset.rotate_and_offset(w_2,h_2,s,c,mean_x,mean_y)
        bottom_left = Dataset.rotate_and_offset(-w_2,h_2,s,c,mean_x,mean_y)
        return [top_left,top_right,bottom_right,bottom_left]
    
    @staticmethod
    def rotate_and_offset(x, y, s, c, mx, my):
        return [x * c - y * s + mx, x * s + y * c + my]
        
    def __init__(self, datatype=1):
        datatype = "train" if datatype else "test"
        datapath = Dataset.data_set_path+"{}/".format(datatype)
        base_metadata = {'source': 'MSRATD500','tags': ["sample", "MSRATD500", "MSRATD500.{}".format(datatype)] }
        default_annotation = {"annotation_tags" : ["annotated.by.MSRATD500"]}
        confidence = 1
        annotation_domain = "text:line"
        
        self.image_metadata = dict()
        # extract information from MSRA TD500 datasets
        # based on https://github.com/blindsightcorp/rigor/blob/master/python/examples/import-MSRATD500.py
        for truthfile in glob.glob("{}*.gt".format(datapath)):
            imagefile = truthfile.rsplit('.', 1)[0] + '.JPG'
            local_image_path = os.path.split(imagefile)
            metadata = dict(base_metadata)
            annotations = list()
            with open(truthfile,'r') as truth:
                for row in truth:
                    (index,difficulty,x,y,w,h,rads) = string.split(row.rstrip())
                    rect = Dataset.rotated_rect(int(x),int(y),int(w),int(h),float(rads))
                    annotation = {"domain" : annotation_domain, "confidence":confidence }
                    annotation_tags = copy.deepcopy(default_annotation)
                    difficulty = "difficulty.standard" if difficulty == "0" else "difficulty.hard"
                    annotation_tags['annotation_tags'].append(difficulty)
                    annotation.update(annotation_tags)
                    annotation.update({'boundary':rect})
                    annotation.update({'difficulty':difficulty})
                    annotations.append(annotation)
            metadata.update({"annotations":annotations})
            image_name = local_image_path[1]
            metadata.update({"source_id":image_name})
            metadata.update({"file_path":imagefile})
            # store it in the result in a metadata map
            self.image_metadata[image_name.rsplit('.', 1)[0]] = metadata
    
    def show_truth(self, image_id):
        metadata = self.image_metadata[image_id]
        print metadata["file_path"]
        image = cv2.imread(metadata["file_path"])
        # draw a box around each annotated text
        for annotation in metadata["annotations"]:
            boundary = annotation["boundary"]
            print boundary
            points = np.array(boundary, np.int32)
            points = points.reshape((-1,1,2))
            print points
            cv2.polylines(image, np.int32([points]), True,(0,255,255))
        show_image(image)

    @staticmethod
    def load_data(filepath):
        # TODO loads saved trained data from filepath
        return None
       
    @staticmethod
    def save_data():
        # TODO saves data to file after training 
        filepath = ""
        return filepath

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path", help="The path to the image")
    parser.add_argument("-t", "--trained", help="File location of trained data")
    args = vars(parser.parse_args())
    
    logging.getLogger("").setLevel(logging.getLevelName("INFO"))    
    
    image = cv2.imread(args["img_path"])
    trained_path = args["trained"] 

    detector = cv2.FeatureDetector_create('MSER')
    points = detector.detect(image)
    
    
    # character candidates extraction
    # uses maximally stable extremal region extractor
    # then prunes according to secotion 3.1 of 
    # Robust Text Detection in Natural Scene Images (Xu-Cheng Yin et al.)
    mser_regions = TextDetector.run_MSER(image, show=True)
    trained_data = None
    if not trained_path:
        # load the dataset 
        training = Dataset(Dataset.TRAINING)
        test = Dataset(Dataset.TEST)
        
        # example for using the dataset
        # training.show_truth("IMG_0582")    
        
        # TODO
        filepath = Dataset.save_data()
        print "Saved trained data to : " , filepath
        trained_data = None
    else:
        trained_data = Dataset.load_data(trained_path)
    
    # filter the character candidates according to
    # section 4 (4.2) of Multi-Orientation Scene Text Detection with Adaptive Clustering (Xu-Cheng Yin et al.)
    candidates = TextDetector.candidates_selection(mser_regions)
    candidates = TextDetector.candidates_elimination(candidates)
    
