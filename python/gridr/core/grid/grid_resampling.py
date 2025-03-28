# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of GRIDR
# (see https://gitlab.cnes.fr/gridr/gridr).
#
#
"""
Grid resampling
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
        py_array1_grid_resampling_f64_u8,
        )

F64_U8_F64_F64 = (np.dtype('float64'), np.dtype('uint8'), np.dtype('float64'), np.dtype('float64'))

PY_ARRAY_GRID_RESAMPLING_FUNC = {
    F64_U8_F64_F64: py_array1_grid_resampling_f64_u8,
}
    
def array_grid_resampling(
        array_in: np.ndarray,
        grid_row: np.ndarray,
        grid_col: np.ndarray,
        grid_resolution: Tuple[int, int],
        array_out: np.ndarray,
        nodata_out: [Union[int, float]] = 0,
        win: Optional[np.ndarray] = None,
        array_in_mask: Optional[np.ndarray] = None,
        grid_mask: Optional[np.ndarray] = None,
        array_out_mask: Optional[np.ndarray] = None,
        ) -> NoReturn:
    """
    
    This methods wraps the rust function 'py_array1_grid_resampling_*'
    
    Args:
        TODO
        
        win: An optional window defined in the full resolution grid coordinate
             system in order to limit the production to a target region in the
             grid. Given that coordinates of the window are given considering
             the full resolution grid, please note that the window limits may
             not match with points in the low resolution input grid.
             The window is given as a list of tuple containing the first and
             last index for each dimension.
                e.g. for a dimension 2 : 
                ((first_row, last_row), (first_col, last_col))
             Its dimension is not the same as the dimension of the input array
             `array_in`, but it must match the dimension of the grids ; assuming
             the grid are 2d, the `win` parameter should contain 2 tuples (as
             shown in the example).
             
    Returns:
        None
    """
    assert(array_in.flags.c_contiguous is True)
    assert(grid_row.flags.c_contiguous is True)
    assert(grid_col.flags.c_contiguous is True)
    assert(array_out.flags.c_contiguous is True)
    
    array_in_shape = array_in.shape
    if len(array_in_shape) == 2:
        array_in_shape = (1, ) + array_in_shape
    array_in = array_in.reshape(-1)
    
    assert(np.all(grid_row.shape == grid_col.shape))
    assert(len(grid_row.shape) == 2)
    grid_shape = grid_row.shape
    grid_row = grid_row.reshape(-1)
    grid_col = grid_col.reshape(-1)
    
    array_out_shape = array_out.shape
    if len(array_out_shape) == 2:
        array_out_shape = (1,) + array_out_shape
    # check same number of variables in array (first dim)
    assert(array_out_shape[0] == array_in_shape[0])
    array_out = array_out.reshape(-1)
    
    array_in_mask_dtype = np.dtype('uint8')
    if array_in_mask is not None:
        array_in_mask_dtype = array_in_mask.dtype
    
    func_types = (array_in.dtype, array_in_mask_dtype, array_out.dtype,
            grid_row.dtype)
    
    py_grid_win = None
    if win is not None:
        py_grid_win = PyArrayWindow2(start_row=win[0][0], end_row=win[0][1],
                start_col=win[1][0], end_col=win[1][1])
    
    nodata_out = array_out.dtype.type(nodata_out)
            
    try:
        func = PY_ARRAY_GRID_RESAMPLING_FUNC[func_types]
    except KeyError:
        raise Exception("py_array_grid_resampling_ function not available for "
                f"types {func_types}")
    else:
        func( array_in=array_in,
                array_in_shape=array_in_shape,
                grid_row=grid_row,
                grid_col=grid_col,
                grid_shape=grid_shape,
                grid_resolution=grid_resolution,
                array_out=array_out,
                array_out_shape=array_out_shape,
                nodata_out=nodata_out,
                array_in_mask=None,
                grid_mask=None,
                array_out_mask=None,
                grid_win=py_grid_win,)