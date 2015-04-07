#!/usr/bin/python

import cv2
import numpy as np
import argparse


class TextDetector(object):
    @staticmethod
    def show(image, window_name="Image"):
        cv2.imshow(window_name, image)
        cv2.waitKey()
        cv2.destroyAllWindows()
        
    @staticmethod    
    def MSER(image, show=False):
        mser = cv2.MSER()
        mser_regions = mser.detect(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)
        # prune the excess regioins
        mser_regions = TextDetector.prune_MSERs(mser_regions)
        if show:
            hulls = [cv2.convexHull(p.reshape(-1,1,2)) for p in mser_regions ]
             # display the image
            display = image.copy()
            cv2.polylines(display, hulls, 1, (0,255,0))
            TextDetector.show(display)
        return mser_regions
    
    @staticmethod
    def prune_MSERs(mser_regions):
        #TODO
        return mser_regions
        
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
        candidates = filter(lambda candidate: good_candidate(candidate), candidates)
        return candidates
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path", help="The path to the image")
    args = vars(parser.parse_args())
    
    image = cv2.imread(args["img_path"])

    # character candidates extraction
    # uses maximally stable extremal region extractor
    # then prunes according to secotion 3.1 of 
    # Robust Text Detection in Natural Scene Images (Xu-Cheng Yin et al.)
    mser_regions = TextDetector.MSER(image, True)
    
    # filter the character candidates according to
    # section 4 of Multi-Orientation Scene Text Detection with Adaptive Clustering (Xu-Cheng Yin et al.)
    candidates = TextDetector.candidates_selection(mser_regions)
    candidates = TextDetector.candidates_elimination(candidates)
    