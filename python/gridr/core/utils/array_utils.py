# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of GRIDR
# (see https://gitlab.cnes.fr/gridr/gridr).
#
#
"""
Array utils module
"""
# pylint: disable=C0413
import sys
PY311 = sys.version_info >= (3,11)
from typing import Tuple, NoReturn
if PY311:
    from typing import Self
else:
    from typing_extensions import Self
import numpy as np
import rasterio
# pylint: enable=C0413

from gridr.cdylib import (
        py_array1_replace_i8,
        py_array1_replace_u8,)

def array_replace(
        array: np.ndarray,
        val_cond: int,
        val_true: int,
        val_false: int,
        ) -> NoReturn:
    """
    
    This methods wraps the rust function 'py_array1_replace_*' for integer types
    
    Args:
        array : array into which elements are replaced inplace
        val_cond : condition values
        val_true : replaced value for elements whose input value equals to 
                'val_cond'
        val_false : replaced value for elements whose input value does not equals
                to 'val_cond'
    
    Returns:
        None
    """
    assert(array.dtype in (np.int8, np.uint8))
    assert(array.flags.c_contiguous is True)
    array = array.reshape(-1)
    if array.dtype == np.int8:
        py_array1_replace_i8(array, val_cond, val_true, val_false)
    elif array.dtype == np.uint8:
        py_array1_replace_u8(array, val_cond, val_true, val_false)
        


class ArrayProfile(object):
    """
    A class to define array attribute. This class aims to be used to mock
    a numpy array object in order to access its attributes members such as
    ndim, shape, dtype and size.
    """
    def __init__(self, shape: Tuple[int], ndim: int, dtype: np.dtype):
        """
        Constructor
        
        Args:
            shape: array shape
            ndim: array number of dimensions
            dtype: array data type
        """
        self.ndim = ndim
        self.shape = shape
        self.dtype = dtype
        self.size = np.prod(self.shape)
        
    @classmethod
    def from_dataset(cls, ds: rasterio.io.DatasetReader) -> Self:
        """
        Method to create an object from a rasterio Dataset.
        
        Args:
            ds: a rasterio dataset reader object (from rasterio.open)
            
        Returns:
            the instanciated ArrayProfile object
        """
        shape = (ds.count, ds.height, ds.width)
        ndim = 3
        if ds.count == 1:
            shape = (ds.height, ds.width)
            ndim = 2
        return cls(
                shape=shape,
                ndim=ndim,
                dtype=np.dtype(ds.profile['dtype']))
