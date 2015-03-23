#! /usr/bin/python

# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__ = "neduchal"
__date__ = "$23.3.2015 9:43:52$"

import numpy as np
import cv2
import directories


class ImageDirectory:

    def __init__(self, path):
        self.dir = directories.directory(path)
                              
        pass
    
    def frame(self, gray = False):
        if self.dir.position < self.dir.getCount():
            if gray:
                image = cv2.imread(self.dir.getNext(),0)
            else:
                image = cv2.imread(self.dir.getNext())
            return image
        else:
            return None