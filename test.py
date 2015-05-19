#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import features
import cv2

#img = cv2.imread('/home/neduchal/Dokumenty/projekty/statistika/data/4.1.06.pgm',0)

#features.init_new_feature(0, 0, img, {'size':[img.shape[1],img.shape[0]]}, {'type':'FAST'}, [])

feat = []
print len(feat)
x = np.zeros(13)
features.add_feature_information(feat, 1, np.zeros([30,20]), x, 0, 0)

print len(feat)
print feat[0]

print x.shape