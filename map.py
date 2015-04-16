#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
MAP MANAGMENT
---------------------

"""

# import funkcí
import sys
import os.path
import numpy as np

# ziskani cest ke skriptum
path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../"))

# nacteni dalsich soucasti
from models import *
from utils import *
from features import *
