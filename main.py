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
        mser_regions = TextDetector.prune_MSERs(universe, root_node, nodes, components_to_points)
        print mser_regions
        if show:
            hulls = [cv2.convexHull(p.reshape(-1,1,2)) for p in mser_regions ]
             # display the image
            display = image.copy()
            cv2.polylines(display, hulls, 1, (0,255,0))
            show_image(display)
        return mser_regions
    
    @staticmethod
    def prune_MSERs(points, root_node, nodes, component_to_points):
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
            return size(difference(componentDelta, component)) / size(component)
        def to_ER_tree(parent_component, parent_ER):
            
            print parent_component.children
            for child in parent_component.children:
                print type(child)
                child_ER = ER(child)
                child_ER.variation = variation(parent_component, child)
                parent_ER.add_child(child_ER)
                if child.children:
                    to_ER_tree(child, ER(child))
            print "ER ",parent_ER
            
        current_component = root_node
        parent_ER = ER(current_component)
        to_ER_tree(root_node, parent_ER)
              
        print parent_ER
        return 
        
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
    
