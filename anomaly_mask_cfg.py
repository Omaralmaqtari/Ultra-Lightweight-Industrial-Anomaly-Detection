# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 11:56:02 2024

Modified by Omar Al-maqtari
"""
def anomaly_mask_cfg():
    return {
    "bottle": {
        "use_mask"     : True,
        "bg_threshold" : 250,
        "bg_reverse"   : True
    },
    "cable": {
        "use_mask"     : False,
        "bg_threshold" : None,
        "bg_reverse"   : None
    },
    "capsule": {
        "use_mask"     : True,
        "bg_threshold" : 120,
        "bg_reverse"   : True
    },
    "carpet": {
        "use_mask"     : False,
        "bg_threshold" : None,
        "bg_reverse"   : None
    },
    "grid": {
        "use_mask"     : False,
        "bg_threshold" : None,
        "bg_reverse"   : None
    },
    "hazelnut": {
        "use_mask"     : True,
        "bg_threshold" : 40,
        "bg_reverse"   : False
    },
    "leather": {
        "use_mask"     : False,
        "bg_threshold" : None,
        "bg_reverse"   : None
    },
    "metal_nut": {
        "use_mask"     : True,
        "bg_threshold" : 40,
        "bg_reverse"   : False
    },
    "pill": {
        "use_mask"     : True,
        "bg_threshold" : 100,
        "bg_reverse"   : False
    },
    "screw": {
        "use_mask"     : True,
        "bg_threshold" : 130,
        "bg_reverse"   : True
    },
    "tile": {
        "use_mask"     : False,
        "bg_threshold" : None,
        "bg_reverse"   : None
    },
    "toothbrush": {
        "use_mask"     : True,
        "bg_threshold" : 30,
        "bg_reverse"   : False
    },
    "transistor": {
        "use_mask"     : True,
        "bg_threshold" : 90,
        "bg_reverse"   : True
    },
    "wood": {
        "use_mask"     : False,
        "bg_threshold" : None,
        "bg_reverse"   : None
    },
    "zipper": {
        "use_mask"     : True,
        "bg_threshold" : 100,
        "bg_reverse"   : True
    }
}
