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
Grid resampling
"""
# pylint: disable=C0413
import sys
from typing import NoReturn, Optional, Tuple, Union

import numpy as np

from gridr.cdylib import PyArrayWindow2, PyInterpolatorType, py_array1_grid_resampling_f64

PY311 = sys.version_info >= (3, 11)

if PY311:
    from typing import Self  # noqa: E402, F401
else:
    from typing_extensions import Self  # noqa: E402, F401
# pylint: enable=C0413


F64_F64_F64 = (np.dtype("float64"), np.dtype("float64"), np.dtype("float64"))

PY_ARRAY_GRID_RESAMPLING_FUNC = {
    F64_F64_F64: py_array1_grid_resampling_f64,
}

PY_INTERPOLATOR_TYPES = {
    "nearest": PyInterpolatorType.Nearest,
    "linear": PyInterpolatorType.Linear,
    "cubic": PyInterpolatorType.OptimizedBicubic,
}


def array_grid_resampling(
    interp: str,
    array_in: np.ndarray,
    grid_row: np.ndarray,
    grid_col: np.ndarray,
    grid_resolution: Tuple[int, int],
    array_out: Optional[np.ndarray],
    array_out_win: Optional[np.ndarray] = None,
    nodata_out: Optional[Union[int, float]] = 0,
    array_in_origin: Optional[Tuple[float, float]] = (0.0, 0.0),
    win: Optional[np.ndarray] = None,
    array_in_mask: Optional[np.ndarray] = None,
    grid_mask: Optional[np.ndarray] = None,
    grid_mask_valid_value: Optional[int] = 1,
    grid_nodata: Optional[float] = None,
    array_out_mask: Optional[Union[np.ndarray, bool]] = None,
    check_boundaries: bool = True,
) -> Tuple[Union[np.ndarray, NoReturn], Union[np.ndarray, NoReturn]]:
    """Resamples an input array based on target grid coordinates, applying an
    optional bilinear interpolation for low resolution grids.

    The method uses target grid coordinates (`grid_row` and `grid_col`) that
    may represent a lower resolution than the input array. Bilinear
    interpolation is applied internally to compute missing target coordinates.
    The oversampling factor is specified by the `grid_resolution` parameter,
    where a value of 1 indicates full resolution.

    The interpolation method is used through the `interp` parameter.

    This method wraps a Rust function (`py_array1_grid_resampling_*`) for
    efficient resampling.

    Parameters
    ----------
    interp: str
        The interpolator name as string. See PY_INTERPOLATOR_TYPES keys for
        accepted values.

    array_in : np.ndarray
        The input array to be resampled. It must be a contiguous 2D (nrow,
        ncol) or 3D (nvar, nrow, ncol) array.

    grid_row : np.ndarray
        A 2D array representing the row coordinates of the target grid, with
        the same shape as `grid_col`. The coordinates target row positions in
        the `array_in` input array.

    grid_col : np.ndarray
        A 2D array representing the column coordinates of the target grid,
        with the same shape as `grid_row`. The coordinates target column
        positions in the `array_in` input array.

    grid_resolution : Tuple[int, int]
        A tuple specifying the oversampling factor for the grid for rows and
        columns. The resolution value of 1 represents full resolution, and
        higher values indicate lower resolution grids.

    array_out : Optional[np.ndarray]
        The output array where the resampled values will be stored.
        If `None`, a new array will be allocated. The shape of the output array
        is either determined based on the resolution and the input grid or by
        the optional `win` parameter.

    array_out_win : Optional[np.ndarray], default None
        An optional `np.ndarray` that designates the specific area in
        `array_out` to receive the resampled data. If `None`, the method will
        populate a default rectangular region starting from `array_out`'s
        top-left corner. This argument is only considered when `array_out` is
        passed, requiring `array_out` to be large enough to contain
        `array_out_win`.

    nodata_out : Optional[Union[int, float]], default 0
        The value to be assigned to "NoData" in the output array. This value
        is used to fill in missing values where no valid resampling could occur
        or where a mask flag is set.

    array_in_origin : Optional[Tuple[float, float]], default (0., 0.)
        Bias to respectively apply to the `grid_row` and `grid_col` coordinates.
        The operation is performed by the wrapped Rust function. Its primary use
        cases include aligning with alternative grid origin conventions or
        handling situations where the provided `array_in` array corresponds to a
        subregion of the complete source raster.

    win : Optional[np.ndarray], default None
        A window (or sub-region) of the full resolution grid to limit the
        resampling to a specific target region. The window is defined as a list
        of tuples containing the first and last indices for each dimension.
        If `None`, the entire grid is processed.

    array_in_mask : Optional[np.ndarray], default None
        A mask for the input array that indicates which parts of `array_in`
        are valid for resampling. If not provided, the entire input array is
        considered valid.

    grid_mask : Optional[np.ndarray], default None
        An optional integer mask array for the grid. Grid cells corresponding to
        `grid_mask_valid_value` are considered **valid**; all other values
        indicate **invalid** cells and will result in `nodata_out` in the output
        array. If not provided, the entire grid is considered valid. The grid
        mask must have the same shape as `grid_row` and `grid_col`.

    grid_mask_valid_value : Optional[int], default 1
        The value in `grid_mask` that designates a **valid** grid cell.
        All values in `grid_mask` that differ from this will be treated as
        **invalid**. This parameter is required if `grid_mask` is provided.

    grid_nodata : Optional[float], default None
        The value in `grid_row` and `grid_col` to consider as **invalid**
        cells. Please note this option is exclusive with `grid_mask`. The
        exclusivity is managed within the bound core method.

    array_out_mask : Optional[Union[np.ndarray, bool]], default None
        A mask for the output array that indicates where the resampled values
        should be stored. If `True`, a new array will be allocated and initially
        filled with 0. The shape of this output mask array is consistent with
        the `array_out` shape. If `None` or not `True`, the entire output array
        is assumed to be valid.

    check_boundaries : bool, default True
        Force a check at each iteration to ensure that the requreid data to
        perform interpolation is available in the source data.
        This parameter can be set to False for performance gain if you are sure
        that all the required data is available.

    Returns
    -------
    Tuple[Union[np.ndarray, NoReturn], Union[np.ndarray, NoReturn]]
        A tuple containing:

        -   The resampled array. If `array_out` was provided, this will be
            `None` (as the result is written in-place).
        -   The resampled output mask. If `array_out_mask` was `False` or
            `None`, this will be `None`.

    Raises
    ------
    Exception
        If the `py_array_grid_resampling_*` function (the underlying Rust
        binding) is not available for the provided input types.

    Notes
    -----

    -   This method is designed for resampling raster-like data using a grid of
        target coordinates.
    -   This method is designed to be embedded in code that works on tiles,
        supporting both tiled inputs and outputs.
    -   For correct results, ensure that the `grid_row` and `grid_col` values
        represent the desired target grid coordinates within the full resolution
        grid system.

    Limitations
    -----------

    -   The method assumes that all input arrays (`array_in`, `grid_row`,
        `grid_col`, etc.) are C-contiguous. If any are not, the method may
        raise an assertion error.
    -   The method assumes that the grid-related arrays (`grid_row`, `grid_col`,
        `grid_mask`) have the same shapes. Mismatched shapes will raise an
        assertion error.
    -   The `win` parameter, if provided, must be compatible with the resolution
        of the grid. If `win` exceeds the bounds of the grid, an error may
        occur.
    -   The method does not handle invalid or missing values in the input arrays
        or masks beyond what's specified by `grid_mask` or `grid_nodata`.
        Users are responsible for ensuring any invalid or missing data is
        appropriately handled before calling the method.
    -   For large grids or arrays, performance may degrade. Users should test
        the method's efficiency for their specific data sizes before using it
        in production.
    -   This method assumes that the input grid is in a "full resolution" grid
        coordinate system. If the coordinate system is different, the resampling
        may produce incorrect results.

    Example
    -------

    .. code-block:: python

        # Example usage with a 2D input array and grid:
        array_in = np.random.rand(100, 100)
        grid_row = np.linspace(0, 99, 50)
        grid_col = np.linspace(0, 99, 50)
        grid_resolution = (2, 2)
        array_out = None
        result, _ = array_grid_resampling(
            interp="cubic",
            array_in=array_in,
            grid_row=grid_row,
            grid_col=grid_col,
            grid_resolution=grid_resolution,
            array_out=array_out
        )

    """
    ret = None
    ret_mask = None

    interp_type = None
    try:
        interp_type = PY_INTERPOLATOR_TYPES[interp]
    except KeyError as err:
        raise Exception(f"Unknown interpolator {interp!r}") from err

    assert array_in.flags.c_contiguous is True
    assert grid_row.flags.c_contiguous is True
    assert grid_col.flags.c_contiguous is True

    array_in_shape = array_in.shape
    if len(array_in_shape) == 2:
        array_in_shape = (1,) + array_in_shape
    array_in = array_in.reshape(-1)

    assert np.all(grid_row.shape == grid_col.shape)
    assert len(grid_row.shape) == 2
    grid_shape = grid_row.shape
    grid_row = grid_row.reshape(-1)
    grid_col = grid_col.reshape(-1)

    py_grid_win = None
    if win is not None:
        py_grid_win = PyArrayWindow2(
            start_row=win[0][0], end_row=win[0][1], start_col=win[1][0], end_col=win[1][1]
        )

    # Allocate array_out if not given
    if array_out is None:
        if array_out_win is not None:
            # Ignore it
            array_out_win = None
        array_out_shape = None
        if win is not None:
            # Take the output shape from the window defined at full resolution
            array_out_shape = (win[0, 1] - win[0, 0] + 1, win[1, 1] - win[1, 0] + 1)

        else:
            # Take the output shape from the grid at full resolution
            array_out_shape = (
                (grid_shape[0] - 1) * grid_resolution[0] + 1,
                (grid_shape[1] - 1) * grid_resolution[1] + 1,
            )

        # Init the array
        array_out_shape = (array_in_shape[0],) + array_out_shape
        array_out = np.empty(array_out_shape, dtype=np.float64, order="C")
        ret = array_out
    assert array_out.flags.c_contiguous is True

    array_out_shape = array_out.shape
    if len(array_out_shape) == 2:
        array_out_shape = (1,) + array_out_shape
    # check same number of variables in array (first dim)
    assert array_out_shape[0] == array_in_shape[0]
    array_out = array_out.reshape(-1)

    py_array_out_win = None
    if array_out_win is not None:
        py_array_out_win = PyArrayWindow2(
            start_row=array_out_win[0][0],
            end_row=array_out_win[0][1],
            start_col=array_out_win[1][0],
            end_col=array_out_win[1][1],
        )

    # Manage optional input mask
    # array_in_mask_dtype = np.dtype("uint8")
    if array_in_mask is not None:
        # array_in_mask_dtype = array_in_mask.dtype
        # check shape
        assert array_in_mask.dtype == np.dtype("uint8")
        assert array_in_mask.shape[0] == array_in_shape[1]
        assert array_in_mask.shape[1] == array_in_shape[2]
        # reshape
        array_in_mask = array_in_mask.reshape(-1)

    # Manage optional output mask
    if array_out_mask is not None:
        try:
            assert array_out_mask.dtype == np.dtype("uint8")
            assert array_out_mask.shape[0] == array_out_shape[1]
            assert array_out_mask.shape[1] == array_out_shape[2]
            array_out_mask = array_out_mask.reshape(-1)
        except AttributeError:
            # Not None and not a numpy array due to exception
            # Test if True
            if array_out_mask is True:
                array_out_mask = np.zeros(array_out_shape[1:], dtype=np.uint8, order="C").reshape(
                    -1
                )
                ret_mask = array_out_mask
            else:
                array_out_mask = None

    func_types = (array_in.dtype, array_out.dtype, grid_row.dtype)

    nodata_out = array_out.dtype.type(nodata_out)

    # Manage grid_mask
    if grid_mask is not None:
        # grid mask must be c-contiguous
        assert grid_mask.flags.c_contiguous is True
        # grid mask must be encoded as unsigned 8 bits integer
        assert grid_mask.dtype == np.dtype("uint8")
        # grid mask shape must be the same has the grids
        assert np.all(grid_mask.shape == grid_shape)
        # Lets flat the grid mask view
        grid_mask = grid_mask.reshape(-1)

    try:
        func = PY_ARRAY_GRID_RESAMPLING_FUNC[func_types]
    except KeyError as err:
        raise Exception(
            f"py_array_grid_resampling_ function not available for types {func_types}"
        ) from err
    else:
        func(
            interp=interp_type,
            array_in=array_in,
            array_in_shape=array_in_shape,
            grid_row=grid_row,
            grid_col=grid_col,
            grid_shape=grid_shape,
            grid_resolution=grid_resolution,
            array_out=array_out,
            array_out_shape=array_out_shape,
            nodata_out=nodata_out,
            array_in_origin=array_in_origin,
            array_in_mask=array_in_mask,
            grid_mask=grid_mask,
            grid_mask_valid_value=grid_mask_valid_value,
            grid_nodata=grid_nodata,
            array_out_mask=array_out_mask,
            grid_win=py_grid_win,
            out_win=py_array_out_win,
            check_boundaries=check_boundaries,
        )
    if ret is not None:
        ret = ret.reshape(array_out_shape).squeeze()
    if ret_mask is not None:
        ret_mask = ret_mask.reshape(array_out_shape[1:])
    return ret, ret_mask
