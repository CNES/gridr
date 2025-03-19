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
from typing import Tuple, NoReturn, Optional, Union
if PY311:
    from typing import Self
else:
    from typing_extensions import Self
import numpy as np
import rasterio
# pylint: enable=C0413

from gridr.cdylib import (
        PyArrayWindow2,
        py_array1_replace_i8,
        py_array1_replace_f32_i8,
        py_array1_replace_f64_i8,
        py_array1_replace_u8,
        py_array1_replace_f32_u8,
        py_array1_replace_f64_u8,
        )

def array_replace(
        array: np.ndarray,
        val_cond: Union[int, float],
        val_true: Union[int, float],
        val_false: Union[int, float],
        array_cond: Optional[np.ndarray] = None,
        array_cond_val: Optional[Union[int, float]] = None,
        win: Optional[np.ndarray] = None,
        ) -> NoReturn:
    """
    
    This methods wraps the rust function 'py_array1_replace_*'
    
    Args:
        array : array into which elements are replaced inplace
        val_cond : condition values
        val_true : replaced value for elements whose input value equals to 
                'val_cond'
        val_false : replaced value for elements whose input value does not equals
                to 'val_cond'
        array_cond : optional array on which to apply condition
        array_cond_val : condition value to use if array_cond is defined
        win: The window is given as a list of tuple
                containing the first and last index for each dimension.
                e.g. for a dimension 2 : 
                ((first_row, last_row), (first_col, last_col))
             its dimension is not the same as the dimension of the input array :
             the number of tuple (first, last) must be equal to the number of
             dimensions in the array.
    Returns:
        None
    """
    assert(array.dtype in (np.int8, np.uint8, np.float32, np.float64))
    assert(array.flags.c_contiguous is True)
    if array_cond is not None:
        assert(array_cond.dtype in (np.int8, np.uint8))
        assert(array_cond_val is not None)
    if array_cond_val is not None:
        assert(array_cond is not None)
        
    py_window = None
    if win is not None:
        py_window = PyArrayWindow2(start_row=win[0][0], end_row=win[0][1],
                start_col=win[1][0], end_col=win[1][1])
    
    nrow, ncol = array.shape
    array = array.reshape(-1)
    if array_cond is not None:
        array_cond = array_cond.reshape(-1)
    
    py_array_replace_func = {
            (np.dtype('int8'), np.dtype('int8')) : py_array1_replace_i8,
            (np.dtype('float32'), np.dtype('int8')) : py_array1_replace_f32_i8,
            (np.dtype('float64'), np.dtype('int8')) : py_array1_replace_f64_i8,
            (np.dtype('uint8'), np.dtype('uint8')) : py_array1_replace_u8,
            (np.dtype('float32'), np.dtype('uint8')) : py_array1_replace_f32_u8,
            (np.dtype('float64'), np.dtype('uint8')) : py_array1_replace_f64_u8,
            }
    if array_cond is not None:
        py_array_replace_func[(array.dtype, array_cond.dtype)](
                array, nrow, ncol, val_cond, val_true, val_false, array_cond,
                array_cond_val, py_window)
    else:
        py_array_replace_func[(array.dtype, array.dtype)](
                array, nrow, ncol, val_cond, val_true, val_false, None, None,
                py_window)


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
