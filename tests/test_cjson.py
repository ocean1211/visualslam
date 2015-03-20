#! /usr/bin/python

# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

import unittest
import os.path
import sys

path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../"))

from inout import cjson

class TestCJson(unittest.TestCase):

    def test_submit(self):
        data = cjson.load(os.path.join(path_to_script, "data/cjson.json"))
        print data["name"]
        self.assert_(data["name"] == "test")

#if __name__ == "__main__":
#    unittest.main()
