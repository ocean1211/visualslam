#! /usr/bin/python
# -*- coding: utf-8 -*-

# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__ = "neduchal"
__date__ = "$20.3.2015 9:02:25$"

import json

def load(filename):
    """
        Načtení komentovaného JSON souboru
        
        :param filename: cesta k souboru
        :type filename: string
    """
    
    fp = open(filename, 'r')
    
    lines = fp.readlines()
    data = "";            
    for line in lines:
        linestrip = line.lstrip()
        if linestrip[0:1] == '#':
            continue
        else:
            data = data + line 

    data = json.loads(data)

    fp.close()
    return data

