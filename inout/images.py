#! /usr/bin/python

# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__ = "neduchal"
__date__ = "$23.3.2015 9:43:52$"

import cv2
import glob


class ImageDirectory:

    def __init__(self, path):
        self.fns = glob.glob(path)
        self.fns = sorted(self.fns)
        self.counter = 0
        pass
    
    def frame(self, gray=False):
        print self.counter, len(self.fns)
        if self.counter < len(self.fns):
            if gray:
                image = cv2.imread(self.fns[self.counter], 0)
            else:
                image = cv2.imread(self.fns[self.counter])
            self.counter += 1
            return image
        else:
            return None