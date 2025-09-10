# coding: utf8
#
# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of GRIDR
# (see https://github.com/CNES/gridr).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
    """Replaces elements within an array in-place based on specified conditions.

    This method is a Python wrapper around the Rust function 
    `py_array1_replace_*`, designed for efficient in-place modification of NumPy
    arrays. It allows conditional replacement based on either the `array` itself
    or an optional `array_cond` (condition array).

    Parameters
    ----------
    array : numpy.ndarray
        The array into which elements are replaced in-place.
        Must be C-contiguous and have a `dtype` of `int8`, `uint8`, `float32`, 
        or `float64`.
        
    val_cond : int or float
        The primary condition value. Elements in `array` (or `array_cond` if 
        provided) equal to `val_cond` will be affected.
        
    val_true : int or float
        The value to replace elements that satisfy the condition (i.e., input 
        value equals `val_cond`).
        
    val_false : int or float
        The value to replace elements that do *not* satisfy the condition (i.e.,
        input value does not equal `val_cond`).
        
    array_cond : numpy.ndarray, optional
        An optional 1D or 2D array on which to apply the condition. If provided,
        the replacement in `array` is based on the values in `array_cond`. Must
        have a `dtype` of `int8` or `uint8`. Defaults to `None`, in which case
        the `array` itself is used for the condition.
        
    array_cond_val : int or float, optional
        The condition value to use if `array_cond` is defined. This value is
        compared against elements in `array_cond`. This parameter is required
        if `array_cond` is provided. Defaults to `None`.
        
    win : numpy.ndarray, optional
        A window `win` to restrict the operation to a specific region of the 
        `array`.
        This is a 2D NumPy array where each row represents a dimension and 
        contains `(min_idx, max_idx)`. **Both `min_idx` and `max_idx` are 
        inclusive**, adhering to the GridR's "window" convention. The window's 
        dimensions must match those of the `array`. Defaults to `None`, meaning 
        the operation applies to the entire array.

    Returns
    -------
    NoReturn
        This function modifies the `array` in-place and does not return any 
        value.

    Raises
    ------
    AssertionError
        If `array`'s `dtype` is not one of `int8`, `uint8`, `float32`, or 
        `float64`.
        
    AssertionError
        If `array` is not C-contiguous (`array.flags.c_contiguous` is `False`).
        
    AssertionError
        If `array_cond` is provided and its `dtype` is not `int8` or `uint8`.
        
    AssertionError
        If `array_cond` is provided but `array_cond_val` is `None`.
        
    AssertionError
        If `array_cond_val` is provided but `array_cond` is `None`.
        
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
    A class to define array attributes for mocking or descriptive purposes.

    This class is designed to hold essential attributes of a NumPy array, such
    as its shape, number of dimensions, data type, and total size, allowing
    access to these members similar to a `numpy.ndarray` object without
    instantiating a full array.
    """
    def __init__(self, shape: Tuple[int, ...], ndim: int, dtype: np.dtype):
        """
        Initializes an `ArrayProfile` object.

        Parameters
        ----------
        shape : tuple of int
            The shape of the array, e.g., `(rows, cols, bands)`.
            
        ndim : int
            The number of dimensions of the array.
            
        dtype : numpy.dtype
            The data type of the array's elements, e.g., `np.int16`, 
            `np.float32`.

        Returns
        -------
        None
            This constructor initializes the object's attributes.
        """
        self.ndim = ndim
        self.shape = shape
        self.dtype = dtype
        self.size = np.prod(self.shape)
        
    @classmethod
    def from_dataset(cls, ds: rasterio.io.DatasetReader) -> Self:
        """
        Creates an `ArrayProfile` object from a `rasterio.io.DatasetReader`.

        This class method extracts relevant array attributes (shape, number of
        dimensions, and data type) from an opened Rasterio dataset. It adjusts
        the `ndim` and `shape` for single-band datasets to represent them as 2D
        arrays, consistent with typical image processing.

        Parameters
        ----------
        ds : rasterio.io.DatasetReader
            A Rasterio dataset reader object, typically obtained via `
            rasterio.open()`.

        Returns
        -------
        ArrayProfile
            An instantiated `ArrayProfile` object populated with attributes
            derived from the provided Rasterio dataset.
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
