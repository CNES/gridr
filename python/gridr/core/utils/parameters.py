# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of GRIDR
# (see https://gitlab.cnes.fr/gridr/gridr).
#
#
"""
Parameters operations utils module
"""
from typing import Union, Tuple, Any

def tuplify(p: Union[Any, Tuple], ndim: int):
    """Util method to convert a single parameter to a tuple.
    If the parameter p is already a sequence (list or tuple) it is returned as
    given.
    Otherwise, the output tuple correspond to the repeated couple (p, p) along
    side each dimension.
    
    e.g : tuplify('a', 3) will return (('a','a'), ('a','a'), ('a','a'))
    
    Args:
        p : the parameter to tuplify
        
    Returns:
        the tuplified parameter
    """
    out = p
    try:
        p[0]
    except TypeError:
        out = ((p, p), ) * ndim
    return out
