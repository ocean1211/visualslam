#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
RANSAC
---------------------

"""

# importy
import numpy as np

# trida rozsireneho kalmanova filtru
class ransac:

    # konstruktor
    def __init__(self): 
        pass
		
    def hypotheses(self, x_p, P_p, features, inparams):
        pass
    
    def matching(self, frame, features, inparams):
        pass
    
    def select_random_match(self, zi, features, numICMatches):
        pass
    
    def search_IC_matches(self, x_p, P_p, map, inparams, frame): 
        pass
    
    def count_matches_under_a_threshold(self, features):
        pass
    
    def compute_hypothesis_support_fast(pos_id, pos_XYZ, x, inparams, state_vector_pattern, zID, zXYZ, features, thresh):
        pass
    
    def rescue_hi_inliers(self, features, map, inparams):
        pass
    
    def update_hi_inliers(self, features): 
        pass

    def update_li_inliers(self, features):
        pass    
        
