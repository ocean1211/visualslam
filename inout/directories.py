#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
IO
---------------------

"""

# Imports
import sys
import os.path
import numpy as np
import scipy
import cv2

# nastaveni slozek, ve kterych se budou hledat funkce
#path_to_script = os.path.dirname(os.path.abspath(__file__))

## directory class
class directory:

        # construktor
        def __init__(self, dirname): #
            self.position = 0
            self.count = 0

            self.filelist = []
            for root, directories, files in os.walk(dirname):
              for filename in files:
                filepath = os.path.join(root, filename)
                self.filelist.append(filepath)
            self.filelist.sort()
            self.count = len(self.filelist)

        def getNext(self):
                self.position = self.position + 1
                return self.filelist[self.position - 1]
                
        def getPrevious(self):
                self.position = self.position - 1
                return self.filelist[self.position + 1]

        def getFirst(self):
                return self.filelist[0]

        def getLast(self):
                return self.filelist[self.count-1]

        def getCount(self):
                return self.count

        pass
