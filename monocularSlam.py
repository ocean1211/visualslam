#! /usr/bin/python

# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

import ekf

class slam():
    
    # konstruktor
    def __init__(self, inputobj, params):     
        self.input = inputobj
        self.counter = 0
        self.params = params
        self.stepOn = true
        self.frame = self.input.frame(gray = True) 
        pass
		
    def init(self):
        
        # EKF filter init
        filter = ekf.ekf(self.params["dtime"], 0.007, 0.007, 1.0)
        
        filter
        
        
        
        # MAP init
        
        # RANSAC init
        pass
    
    def step(self):

        self.frame = self.input.frame(gray = True) 
        
        if (self.stepOn == False):
            return -1
        
        # MAP.manage
        
        # Filter prediction
        
        # Ransac
        
        # Filter update
        
        # show
        pass