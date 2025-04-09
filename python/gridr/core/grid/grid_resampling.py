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
        array_out: Optional[np.ndarray],
        nodata_out: [Union[int, float]] = 0,
        win: Optional[np.ndarray] = None,
        array_in_mask: Optional[np.ndarray] = None,
        grid_mask: Optional[np.ndarray] = None,
        array_out_mask: Optional[np.ndarray] = None,
        ) -> Union[np.ndarray, NoReturn]:
    """
    Resamples an input array based on target grid coordinates, applying an
    optional bilinear interpolation for low resolution grids.

    The method uses target grid coordinates (`grid_row` and `grid_col`) that may
    represent a lower resolution than the input array. Bilinear interpolation is
    applied internally to compute missing target coordinates. The oversampling
    factor is specified by the `grid_resolution` parameter, where a value of 1 
    indicates full resolution.

    This method wraps a Rust function (`py_array1_grid_resampling_*`) for
    efficient resampling.

    Args:
    -----
        array_in : np.ndarray
            The input array to be resampled. It must be a contiguous 2D (nrow, 
            ncol) or 3D (nvar, nrow, ncol) array.

        grid_row : np.ndarray
            A 2D array representing the row coordinates of the target grid, with
            the same shape as `grid_col`. The coordinates targets row positions 
            in the `array_in` input array.

        grid_col : np.ndarray
            A 2D array representing the column coordinates of the target grid,
            with the same shape as `grid_row`. The coordinates targets column
            positions in the `array_in` input array.

        grid_resolution : Tuple[int, int]
            A tuple specifying the oversampling factor for the grid for rows and
            columns. The resolution value of 1 represents full resolution, and
            higher values indicate lower resolution grids.

        array_out : Optional[np.ndarray]
            The output array where the resampled values will be stored.
            If `None`, a new array will be allocated. The shape of the output
            array is either determined based on the resolution and the input
            grid or by the optional `win` parameter.

        nodata_out : Union[int, float], default 0
            The value to be assigned to "NoData" in the output array. This value
            is used to fill in missing values where no valid resampling could
            occur or where a mask flag is set.

        win : Optional[np.ndarray], default None
            A window (or sub-region) of the full resolution grid to limit the
            resampling to a specific target region. The window is defined as a
            list of tuples containing the first and last indices for each
            dimension.
            If `None`, the entire grid is processed.

        array_in_mask : Optional[np.ndarray], default None
            A mask for the input array that indicates which parts of `array_in`
            are valid for resampling.
            If not provided, the entire input array is considered valid.

        grid_mask : Optional[np.ndarray], default None
            A mask for the grid, where values of `1` represent invalid grid
            cells.
            If not provided, the entire grid is considered valid.
            The grid mask must have the same shape as `grid_row` and `grid_col`.

        array_out_mask : Optional[np.ndarray], default None
            A mask for the output array that indicates where the resampled
            values should be stored. 
            If not provided, the entire output array is assumed to be valid.

    Returns:
    --------
    np.ndarray
        The resampled array, unless `array_out` was provided, in which case
        `None` is returned.

    Raises:
    -------
    Exception
        If the `py_array_grid_resampling_*` function is not available for the
        provided input types.

    Limitations:
    ------------
    - The method assumes that all input arrays (`array_in`, `grid_row`,
     `grid_col`, etc.) are C-contiguous. If any of them are not, the method may
      raise an assertion error.
    - The method assumes that the grid related arrays (`grid_row`, `grid_col`, 
      `grid_mask`) have the same shapes. Mismatched shapes between `grid_row`,
      `grid_col`, and `grid_mask` will raise an assertion error.
    - The `win` parameter, if provided, must be compatible with the resolution
      of the grid. 
      If `win` exceeds the bounds of the grid, an error may occur.
    - The method does not handle invalid or missing values in the input arrays
      or masks. 
      Users are responsible for ensuring that any invalid or missing data is
      appropriately handled before calling the method.
    - For large grids or arrays, performance may degrade. Users should test 
      the method's efficiency for their specific data sizes before using it in
      production.
    - This method assumes that the input grid is in a "full resolution" grid
      coordinate system. 
      If the coordinate system is different, the resampling may produce
      incorrect results.

    Example:
    --------
    # Example usage with a 2D input array and grid:
    array_in = np.random.rand(100, 100)
    grid_row = np.linspace(0, 99, 50)
    grid_col = np.linspace(0, 99, 50)
    grid_resolution = (2, 2)
    array_out = None

    result = array_grid_resampling(
        array_in=array_in, 
        grid_row=grid_row, 
        grid_col=grid_col, 
        grid_resolution=grid_resolution, 
        array_out=array_out
    )

    Notes:
    ------
    - This method is designed for resampling raster-like data using a grid of
      target coordinates.
    - This method is designed so that it can be embedded in a code that works
      on tiles both for the inputs and the outputs.
    - For correct results, ensure that the `grid_row` and `grid_col` values
      represent the desired target grid coordinates in the full resolution grid
      system.
    """
    ret = None
    assert(array_in.flags.c_contiguous is True)
    assert(grid_row.flags.c_contiguous is True)
    assert(grid_col.flags.c_contiguous is True)
    
    array_in_shape = array_in.shape
    if len(array_in_shape) == 2:
        array_in_shape = (1, ) + array_in_shape
    array_in = array_in.reshape(-1)
    
    assert(np.all(grid_row.shape == grid_col.shape))
    assert(len(grid_row.shape) == 2)
    grid_shape = grid_row.shape
    grid_row = grid_row.reshape(-1)
    grid_col = grid_col.reshape(-1)
    
    py_grid_win = None
    if win is not None:
        py_grid_win = PyArrayWindow2(start_row=win[0][0], end_row=win[0][1],
                start_col=win[1][0], end_col=win[1][1])
    
    # Allocate array_out if not given
    if array_out is None:
        array_out_shape = None
        if win is not None:
            # Take the output shape from the window defined at full resolution
            array_out_shape = (win[0,1] - win[0,0] + 1, win[1,1] - win[1,0] + 1)
                    
        else:
            # Take the output shape from the grid at full resolution
            array_out_shape = (
                    (grid_shape[0] - 1) * grid_resolution[0] + 1,
                    (grid_shape[1] - 1) * grid_resolution[1] + 1,)

        # Init the array
        array_out_shape = (array_in_shape[0], ) + array_out_shape;
        array_out = np.empty(array_out_shape, dtype=np.float64, order='C')
        ret = array_out
    assert(array_out.flags.c_contiguous is True)
    
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
    
    nodata_out = array_out.dtype.type(nodata_out)
    
    # Manage grid_mask
    if grid_mask is not None:
        # grid mask must be c-contiguous
        assert(grid_mask.flags.c_contiguous is True)
        # grid mask must be encoded as unsigned 8 bits integer
        assert(grid_mask.dtype == np.dtype('uint8'))
        # grid mask shape must be the same has the grids
        assert(np.all(grid_mask.shape == grid_shape))
        # Lets flat the grid mask view
        grid_mask = grid_mask.reshape(-1)
        
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
                grid_mask=grid_mask,
                array_out_mask=None,
                grid_win=py_grid_win,)
    if ret is not None:
        ret = ret.reshape(array_out_shape).squeeze()
    return ret