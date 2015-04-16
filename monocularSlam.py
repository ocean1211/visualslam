#! /usr/bin/python

# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

import ekf

class slam:    
    # konstruktor
    def __init__(self, inputobj, params):     
        self.input = inputobj
        self.counter = 0
        self.params = params
        self.stepOn = true
        self.frame = self.input.frame(gray=True) 
        pass
		
    def init(self):
        
        # EKF filter init
        filter = ekf.ekf(self.params["dtime"], 0.007, 0.007, 1.0)
        
        v0 = 0;
        w0 = 1.0e-15;
        stdV = 0.025;
        stdW = 0.025;        
        
        filter.x = np.array([0, 0, 0, 1, 0, 0, 0, v0, v0, v0, w0, w0, w0])
        filter.P[0:7, 0:7] = np.identity([7, 7]) * 2.2204e-016
        filter.P[7:10, 7:10] = np.identity([3, 3]) * (stdV ** 2)
        filter.P[10:13, 10:13] = np.identity([3, 3]) * (stdW ** 2)
        
        # MAP init
        
        # RANSAC init
        pass
    
    def step(self):

        self.frame = self.input.frame(gray=True) 
        
        if (self.stepOn == False):
            return -1
        
        # MAP.manage
        
        # Filter prediction
        
        # Ransac
        
        # Filter update
        
        # show
        pass