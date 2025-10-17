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
import logging
import sys
from typing import Any, NoReturn, Optional, Tuple, Union

import numpy as np

from gridr.cdylib import PyArrayWindow2, py_array1_grid_resampling_f64
from gridr.core.grid.grid_commons import grid_full_resolution_shape, grid_resolution_window_safe
from gridr.core.grid.grid_mask import Validity
from gridr.core.grid.grid_utils import (
    array_compute_resampling_grid_geometries,
    array_compute_resampling_grid_src_boundaries,
    read_win_from_grid_metrics,
)
from gridr.core.interp.bspline_prefiltering import array_bspline_prefiltering
from gridr.core.interp.interpolator import (
    Interpolator,
    InterpolatorIdentifier,
    get_interpolator,
    is_bspline,
)
from gridr.core.utils.array_pad import pad_inplace
from gridr.core.utils.array_utils import ArrayProfile

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


STANDALONE_SAFECHECK_SOURCE_BOUNDARIES = True
"""
Parameter activating an additional validation of the grid source boundaries to
ensure topological consistency in standalone mode.

This check computes the source boundaries from all valid grid data within the
current computed region, verifying that the source boundaries extracted from
grid metrics align with the hull border.
When using grid metrics only, we assumes that points inside the source hull
correspond to points within the target hull, maintaining topological integrity.
If this assumption is violated, the read window may be insufficient, potentially
causing a Rust panic when attempting to access out-of-bounds indices.

This safety check helps prevent such runtime errors by proactively extending
boundary conditions if required.
"""


def calculate_source_extent(
    interp: Interpolator,
    array_in: np.ndarray,
    grid_row: np.ndarray,
    grid_col: np.ndarray,
    grid_resolution: Tuple[int, int],
    grid_nodata: Optional[Union[int, float]],
    grid_mask: np.ndarray,
    grid_mask_valid_value: Optional[int] = 1,
    win: Optional[np.ndarray] = None,
    safecheck_src_boundaries: Optional[bool] = True,
    logger_msg_prefix: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
):
    """Calculate the source array read window with margins for interpolation.

    This function computes the minimal read window from the source array required to
    resample data onto the target grid. It accounts for interpolation margins and handles
    boundary cases where the required window extends beyond the source array bounds.

    The calculation proceeds in three steps:

    1. Compute grid metrics from valid grid coordinates
    2. Apply interpolation margins to determine the required source extent
    3. Adjust for boundary conditions and compute necessary padding

    Parameters
    ----------
    interp : Interpolator
        Interpolator instance that defines the required margins for interpolation.
        See `gridr.core.interp.interpolator` for details.

    array_in : np.ndarray
        Source array to be resampled. Must be a C-contiguous 3D array with shape
        (nvar, nrow, ncol) or 2D array with shape (nrow, ncol).

    grid_row : np.ndarray
        2D array of row coordinates in the source array coordinate system.
        Must have the same shape as `grid_col`.

    grid_col : np.ndarray
        2D array of column coordinates in the source array coordinate system.
        Must have the same shape as `grid_row`.

    grid_resolution : tuple of int
        Oversampling factor as (row_resolution, col_resolution). Value of 1 indicates
        full resolution; higher values indicate coarser grids.

    grid_nodata : int or float, optional
        Value in `grid_row` and `grid_col` indicating invalid cells. Mutually
        exclusive with `grid_mask` (exclusivity enforced in core method).

    grid_mask : np.ndarray, optional
        Integer mask array for the grid. Cells matching `grid_mask_valid_value` are
        considered valid. Must have the same shape as `grid_row` and `grid_col`.

    grid_mask_valid_value : int, default=1
        Value in `grid_mask` that designates valid grid cells. Required if
        `grid_mask` is provided.

    win : np.ndarray, optional
        Target window in full-resolution grid coordinates, shape (2, 2):
        ``[[row_start, row_end], [col_start, col_end]]``. If None, processes
        the entire grid.

    safecheck_src_boundaries : bool, default=True
        If True, computes the source boundaries from all valid grid data.

    logger_msg_prefix : str, optional
        Prefix for log messages generated by this function.

    logger : logging.Logger, optional
        Logger instance for diagnostic messages.

    Returns
    -------
    array_src_win_read : np.ndarray, shape (2, 2)
        Final source read window adjusted for margins and boundary constraints.
        Format: ``[[row_start, row_end], [col_start, col_end]]``. Use this window
        for raster IO operations.

    array_src_win_marged : np.ndarray, shape (2, 2)
        Desired read window with margins applied, before boundary correction.
        Format: ``[[row_start, row_end], [col_start, col_end]]``.

    pad : np.ndarray, shape (2, 2)
        Padding required if the marged window extends outside the source array.
        Format: ``[[top_pad, bottom_pad], [left_pad, right_pad]]``.

    Notes
    -----
    - If no valid grid points are found, returns None for all outputs
    - The `safecheck_src_boundaries` option is useful to detect grid topology
      issues that could cause panic
    - Padding may be required when the grid extends beyond source boundaries,
      which should be filled using appropriate boundary conditions

    See Also
    --------
    array_compute_resampling_grid_geometries : Computes grid metrics from coordinates
    read_win_from_grid_metrics : Derives read window from grid metrics
    source_extent_pad : Applies padding to source arrays

    """

    def DEBUG(msg):
        if logger:
            logger.debug(f"{logger_msg_prefix} - {msg}")

    def WARNING(msg):
        if logger:
            logger.warning(f"{logger_msg_prefix} - {msg}")

    # Computing total required margins
    # (top, bottom, left, right)
    margin = np.asarray(interp.total_margins()).reshape((2, 2))
    DEBUG(f"Required margins for interpolator {interp.shortname()} : {margin}")

    # Compute explicit grid target window if it is not defined
    if win is None:
        full_shape_out = grid_full_resolution_shape(
            shape=grid_row.shape, resolution=grid_resolution
        )
        win = np.array(((0, full_shape_out[0] - 1), (0, full_shape_out[1] - 1)))

    # Determine the minimal coarse-grid window containing the oversampled window
    # `oversamped_grid_win`.
    grid_arr_win, _ = grid_resolution_window_safe(
        resolution=grid_resolution, win=win, grid_shape=grid_row.shape
    )

    # Compute strip grid metrics for current tile
    DEBUG("Computing grid metrics... ")
    grid_metrics = array_compute_resampling_grid_geometries(
        grid_row=grid_row,
        grid_col=grid_col,
        grid_resolution=grid_resolution,
        win=grid_arr_win,
        grid_mask=grid_mask,
        grid_mask_valid_value=grid_mask_valid_value,
        grid_nodata=grid_nodata,  # TODO : check not None is supported
    )

    # Grid metrics may be None if no valid points
    if grid_metrics:

        if safecheck_src_boundaries:
            DEBUG("SAFECHECK_SOURCE_BOUNDARIES : Computing source boundaries... ")

            # Compute source boundaries from all valid coordinates
            safe_src_boundaries = array_compute_resampling_grid_src_boundaries(
                grid_row=grid_row,
                grid_col=grid_col,
                win=grid_arr_win,
                grid_mask=grid_mask,
                grid_mask_valid_value=grid_mask_valid_value,
                grid_nodata=grid_nodata,  # TODO : check not None is supported
            )
            DEBUG(f"SAFECHECK_SOURCE_BOUNDARIES : {safe_src_boundaries}")

            # Check that the grid preserve the source topology
            if (
                safe_src_boundaries.xmin < grid_metrics.src_bounds.xmin
                or safe_src_boundaries.xmax > grid_metrics.src_bounds.xmax
                or safe_src_boundaries.ymin < grid_metrics.src_bounds.ymin
                or safe_src_boundaries.ymax > grid_metrics.src_bounds.ymax
            ):
                # Boundaries extend is required !
                WARNING(
                    "SAFECHECK_SOURCE_BOUNDARIES : The grid does not respect the source topology"
                    " - the source boundaries have to be expanded"
                )
                # Replace the source boundaries
                DEBUG("SAFECHECK_SOURCE_BOUNDARIES : Expanding grid metrics source boundaries... ")
                grid_metrics.src_bounds = safe_src_boundaries

        array_src_profile = array_in
        if array_in.ndim == 3:
            array_src_profile = array_in[0]

        array_src_win_read, array_src_win_marged, pad = read_win_from_grid_metrics(
            grid_metrics=grid_metrics,
            array_src_profile_2d=array_src_profile,
            margins=margin,
            logger=logger,
            logger_msg_prefix=logger_msg_prefix,
        )

    return array_src_win_read, array_src_win_marged, pad


def get_array_padded_shape(
    array_src: np.ndarray,
    pad: Tuple[int, int],
) -> (Tuple[int], Tuple[slice]):
    """Compute padded array shape and source window slice for padding operations.

    This utility function calculates the shape of an array after padding and
    generates slice objects to position the original data within the padded array.

    Parameters
    ----------
    array_src : np.ndarray
        Source array to be padded. Must be 2D (nrow, ncol) or 3D (nvar, nrow, ncol).

    pad : tuple of tuple of int
        Padding amounts as ``((top, bottom), (left, right))``.

    Returns
    -------
    array_padded_shape : tuple of int
        Shape of the padded array:

        - For 3D input: (nvar, nrow + top + bottom, ncol + left + right)
        - For 2D input: (nrow + top + bottom, ncol + left + right)

    source_window : tuple of slice
        Slice objects to position the original array within the padded array:

        - For 3D: (slice(None), slice(top, top+nrow), slice(left, left+ncol))
        - For 2D: (slice(top, top+nrow), slice(left, left+ncol))

    Raises
    ------
    ValueError
        If input array has neither 2 nor 3 dimensions.

    Notes
    -----
    This function does not allocate or modify arrays; it only computes metadata
    for padding operations. Use `source_extent_pad` to apply actual padding.

    Examples
    --------
    >>> import numpy as np
    >>> from gridr import get_array_padded_shape
    >>>
    >>> # For a 2D array
    >>> arr = np.ones((100, 100))
    >>> pad = ((5, 5), (10, 10))
    >>> shape, window = get_array_padded_shape(arr, pad)
    >>> print(shape)  # (110, 120)
    >>> print(window)  # (slice(5, 105), slice(10, 110))
    >>>
    >>> # For a 3D array
    >>> arr = np.ones((3, 100, 100))
    >>> shape, window = get_array_padded_shape(arr, pad)
    >>> print(shape)  # (3, 110, 120)
    >>> print(window)  # (slice(None), slice(5, 105), slice(10, 110))

    """
    array_padded_shape, source_window = None, None

    match array_src.ndim:
        case 3:
            array_padded_shape = (
                array_src.shape[0],
                array_src.shape[1] + pad[0][0] + pad[0][1],
                array_src.shape[2] + pad[1][0] + pad[1][1],
            )
            source_window = (
                slice(None, None),
                slice(pad[0][0], pad[0][0] + array_src.shape[1]),
                slice(pad[1][0], pad[1][0] + array_src.shape[2]),
            )
        case 2:
            array_padded_shape = (
                array_src.shape[0] + pad[0][0] + pad[0][1],
                array_src.shape[1] + pad[1][0] + pad[1][1],
            )
            source_window = (
                slice(pad[0][0], pad[0][0] + array_src.shape[0]),
                slice(pad[1][0], pad[1][0] + array_src.shape[1]),
            )
        case _:
            raise ValueError("Input array must have 2 or 3 dimensions")
    return array_padded_shape, source_window


def source_extent_pad(
    array_src: np.ndarray,
    pad,
    boundary_condition,
    fill: Optional[Any] = None,
) -> np.ndarray:
    """Apply padding to a source array with specified boundary conditions.

    This function creates a padded version of the input array, optionally filling
    the padded regions using boundary conditions (edge, reflect, symmetric, wrap)
    or a constant fill value.

    Parameters
    ----------
    array_src : np.ndarray
        Source array to pad. Must be 2D (nrow, ncol) or 3D (nvar, nrow, ncol),
        C-contiguous.

    pad : tuple of tuple of int
        Padding amounts as ``((top, bottom), (left, right))``.

    boundary_condition : str, optional
        Boundary condition mode for padding the margins. If None, padded regions
        are left uninitialized (except if `fill` is provided). Available modes:

        - 'edge': Repeat edge values
        - 'reflect': Mirror reflection without repeating edge
        - 'symmetric': Mirror reflection with repeating edge
        - 'wrap': Circular wrap-around

    fill : scalar, optional
        Value to initialize padded regions before applying boundary conditions.
        If None, array is allocated uninitialized (faster but contains garbage
        values in padded regions if `boundary_condition` is None).

    Returns
    -------
    array_padded : np.ndarray
        Padded array with the same dtype as `array_src`. Shape:

        - For 3D input: (nvar, nrow + top + bottom, ncol + left + right)
        - For 2D input: (nrow + top + bottom, ncol + left + right)

    Notes
    -----
    - The original data is always copied into the center of the padded array
    - If both `boundary_condition` and `fill` are provided, `fill` is applied
      first, then the boundary condition overwrites the padded regions
    - Uses an optimized in-place padding implementation (`pad_inplace`)
    - The returned array is always C-contiguous

    See Also
    --------
    get_array_padded_shape : Computes padded shape without allocation
    pad_inplace : Low-level in-place padding function

    Examples
    --------
    >>> import numpy as np
    >>> from gridr import source_extent_pad
    >>>
    >>> # Pad with edge replication
    >>> arr = np.arange(9).reshape(3, 3)
    >>> padded = source_extent_pad(arr, pad=((1, 1), (1, 1)),
    ...                            boundary_condition='edge')
    >>> print(padded.shape)  # (5, 5)
    >>>
    >>> # Pad with constant fill value
    >>> padded = source_extent_pad(arr, pad=((2, 2), (2, 2)),
    ...                            boundary_condition=None, fill=-999)
    >>>
    >>> # Pad 3D array with reflection
    >>> arr_3d = np.random.rand(3, 100, 100)
    >>> padded_3d = source_extent_pad(arr_3d, pad=((5, 5), (5, 5)),
    ...                               boundary_condition='reflect')
    """
    array_padded_shape, source_window = get_array_padded_shape(array_src, pad)

    # Allocate a new buffer
    array_padded = None
    if fill is not None:
        array_padded = np.full(array_padded_shape, fill, dtype=array_src.dtype, order="C")
    else:
        array_padded = np.empty(array_padded_shape, dtype=array_src.dtype, order="C")

    # Copy original data
    array_padded[source_window] = array_src[:]

    # Apply the boundary condition if any
    if boundary_condition:
        if array_padded.ndim == 3:
            pad = ((0, 0),) + tuple(pad)

        pad_inplace(
            array=array_padded,
            src_win=source_window,
            pad_width=pad,
            mode=boundary_condition,
            strict_size=True,
        )
    return array_padded


def array_grid_resampling(
    interp: InterpolatorIdentifier,
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
    interp_kwargs: Optional[dict] = None,
    standalone: Optional[bool] = True,
    boundary_condition: Optional[bool] = None,
    logger_msg_prefix: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Union[np.ndarray, NoReturn], Union[np.ndarray, NoReturn]]:
    """Resamples an input array based on target grid coordinates, applying an
    optional bilinear interpolation for low resolution grids.

    The method uses target grid coordinates (`grid_row` and `grid_col`) that
    may represent a lower resolution than the input array. Bilinear
    interpolation is applied internally to compute missing target coordinates.
    The oversampling factor is specified by the `grid_resolution` parameter,
    where a value of 1 indicates full resolution.

    The interpolation method is set through the `interp` parameter.

    This method wraps a Rust function (`py_array1_grid_resampling_*`) for
    efficient resampling. The underlying Rust implementation requires that:

    1. All target positions in `array_in` referenced by grid coordinates must be
       accessible, along with their neighborhoods needed for interpolation and
       preprocessing (e.g., B-spline prefiltering).

    2. Any required preprocessing (e.g., B-spline prefiltering) must be completed
       before interpolation.

    **Execution Modes**

    The method supports two execution modes controlled by the `standalone` parameter:

    **Standalone Mode (standalone=True)**:

        Handles all preprocessing automatically, making the function fully self-contained:

        - **Automatic Padding**: If `array_in` is too small to satisfy interpolation
          requirements (e.g., neighborhood access for the interpolator or grid
          coordinates falling near boundaries), the array is automatically padded
          according to `boundary_condition`.

        - **Mask Handling**: If `array_in_mask` is provided, it is padded consistently
          with `array_in`:

          * With `boundary_condition` set: Padded mask values are extrapolated from
            the original mask according to the boundary condition (e.g., 'edge'
            repeats boundary mask values, 'reflect' mirrors them without including the edge).

          * With `boundary_condition=None`: Padded regions are marked as invalid
            (typically set to 0).

          * If no mask is provided and `boundary_condition=None`: A mask is created
            with padded regions marked as invalid.

        - **Preprocessing**: All required preprocessing steps (e.g., B-spline
          coefficient calculation) are performed automatically.

        Use this mode when calling the function independently or when you want
        the function to handle all edge cases automatically.

    **Integrated Mode (standalone=False)**:

        Assumes preprocessing has been handled externally, offering maximum performance:

        - **No Padding**: Assumes `array_in` is already large enough to satisfy all
          interpolation requirements. The caller is responsible for ensuring adequate
          array dimensions.

        - **No Mask Preprocessing**: Assumes `array_in_mask` (if provided) is already
          properly sized and formatted.

        - **No Preprocessing**: Assumes all required preprocessing (e.g., B-spline
          prefiltering) has been completed externally.

        Use this mode within tiled processing pipelines where padding and preprocessing
        are managed at a higher level to avoid redundant operations across tiles.

    Parameters
    ----------
    interp: InterpolatorIdentifier
        The interpolator identifier. It can be:

        - A string representing the interpolator name (e.g., "nearest", "linear"
          , "cubic", etc.).
        - A `PyInterpolatorType` enum value.
        - An instance of an interpolator class.

        See `gridr.core.interp.interpolator` for further details

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
        Force a check at each iteration to ensure that the required data to
        perform interpolation is available in the source data.
        This parameter adresses the core Rust function and can be set to False
        for performance gain if you are sure that all the required data is
        available.

    interp_kwargs : Optional[dict], default=None
        Optional keyword parameters that will be passed to the `get_interpolator`
        function for interpolator creation. Used when `interp` is of type `str`
        or `PyInterpolatorType`.

    standalone : bool, default=True
        Controls the execution mode:

        - `True`: **Standalone mode** - Performs all preprocessing automatically,
          including padding, mask handling, and any required interpolator-specific
          preprocessing (e.g., B-spline prefiltering). Use when calling this
          function independently.

        - `False`: **Integrated mode** - Assumes all preprocessing has been handled
          externally. Offers maximum performance for tiled processing pipelines.
          The caller must ensure `array_in` is adequately sized and preprocessed.

    boundary_condition : Optional[str], default=None
        Defines how to handle boundary conditions when padding is required in
        standalone mode. Ignored when `standalone=False`. Options:

        - `'edge'`: Pad with the edge values of the array (repeat boundary values).
        - `'wrap'`: Wrap around to the opposite edge (circular/periodic boundary).
        - `'reflect'`: Mirror reflection without repeating the edge values.
        - `'symmetric'`: Mirror reflection with edge values repeated.
        - `None`: No padding is applied. If insufficient data is available for
          interpolation, those regions will be marked as invalid in the mask.

        The boundary condition applies to both `array_in` and `array_in_mask`
        (if provided). For masks, the boundary values are extrapolated according
        to the same rule, or marked as invalid if `boundary_condition=None`.

    logger_msg_prefix : Optional[str], default=None
        A prefix to add to all log messages generated by this function.

    logger : Optional[logging.Logger], default=None
        A logger instance for outputting diagnostic messages.

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
    ValueError
        If incompatible parameters are provided (e.g., both `array_in_origin` and
        `standalone=True`).

    AssertionError
        If input arrays are not C-contiguous.

    AssertionError
        If grid-related arrays have mismatched shapes.

    AssertionError
        If optional `array_in_mask` shape is not consistent with `array_in`.

    Exception
        If the `py_array_grid_resampling_*` function (the underlying Rust
        binding) is not available for the provided input types.

    Notes
    -----

    -   This method is designed for resampling raster-like data using a grid of
        target coordinates.
    -   With integrated mode (`standalone=False`) this method is designed to be
        embedded in code that works on tiles, supporting both tiled inputs and
        outputs.
    -   For correct results, ensure that the `grid_row` and `grid_col` values
        represent the desired target grid coordinates within the full resolution
        grid system.
    -   When `standalone=True`, the function may allocate temporary arrays
        internally, which may increase memory usage.

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
    -   For large grids or arrays, performance may degrade. Users should test
        the method's efficiency for their specific data sizes before using it
        in production.
    -   This method assumes that the input grid is in a "full resolution" grid
        coordinate system. If the coordinate system is different, the resampling
        may produce incorrect results.

    Example
    -------

    **Standalone mode with automatic padding**:

    .. code-block:: python

        >>> array_in = np.random.rand(100, 100)
        >>> grid_row = np.linspace(0, 99, 50)
        >>> grid_col = np.linspace(0, 99, 50)
        >>> grid_resolution = (2, 2)
        >>> result, mask = array_grid_resampling(
        ...     interp="cubic",
        ...     array_in=array_in,
        ...     grid_row=grid_row,
        ...     grid_col=grid_col,
        ...     grid_resolution=grid_resolution,
        ...     array_out=None,
        ...     standalone=True,
        ...     boundary_condition='reflect'
        ... )

    **Integrated mode for tiled processing**:

    .. code-block:: python

        >>> # Assume array_in is already padded and preprocessed
        >>> updated_array = preprocess(interp, array_in)  # External function
        >>> result, mask = array_grid_resampling(
        ...     interp="cubic",
        ...     array_in=updated_array,
        ...     grid_row=grid_row,
        ...     grid_col=grid_col,
        ...     grid_resolution=grid_resolution,
        ...     array_out=None,
        ...     standalone=False
        ... )
    """
    ret = None
    ret_mask = None

    # First perform some checks
    assert array_in.flags.c_contiguous is True
    assert grid_row.flags.c_contiguous is True
    assert grid_col.flags.c_contiguous is True

    array_in_shape = array_in.shape
    if len(array_in_shape) == 2:
        array_in_shape = (1,) + array_in_shape

    assert np.all(grid_row.shape == grid_col.shape)
    assert len(grid_row.shape) == 2
    grid_shape = grid_row.shape

    # Manage optional input mask
    # array_in_mask_dtype = np.dtype("uint8")
    if array_in_mask is not None:
        # array_in_mask_dtype = array_in_mask.dtype
        # check shape
        assert array_in_mask.dtype == np.dtype("uint8")
        assert array_in_mask.shape[0] == array_in_shape[1]
        assert array_in_mask.shape[1] == array_in_shape[2]

    # Getting the interpolator object from its identifier
    interp_kwargs = interp_kwargs if interp_kwargs is not None else {}
    interp = get_interpolator(interp, **interp_kwargs)

    if standalone:
        if array_in_origin is not None and np.any(np.asarray(array_in_origin) != 0.0):
            raise ValueError("Shifting the array origin is not available for standalone mode")

        # Initialize the interpolator - required for B-spline for instance
        interp.initialize()

        # Compute require source extent in order to compute resampling from the
        # grid.
        (
            _,
            array_src_win_marged,
            pad,
        ) = calculate_source_extent(
            interp=interp,
            array_in=array_in,
            grid_row=grid_row,
            grid_col=grid_col,
            grid_resolution=grid_resolution,
            grid_nodata=grid_nodata,
            grid_mask=grid_mask,
            grid_mask_valid_value=grid_mask_valid_value,
            win=win,
            safecheck_src_boundaries=STANDALONE_SAFECHECK_SOURCE_BOUNDARIES,
            logger_msg_prefix=logger_msg_prefix,
            logger=logger,
        )

        # Check if input array is sufficient
        if np.any(pad != 0):
            # TODO : not optimal if there is no padding required at an edge,
            # we may not need all input data from the input array.
            # The source_extent_pad method does only perfom padding and do not
            # consider cropping.
            array_padded_fill = None
            mask_padded_fill = None
            mask_padded = None

            if boundary_condition is None:
                array_padded_fill = 0
                mask_padded_fill = Validity.INVALID

            array_padded = source_extent_pad(
                array_src=array_in,
                pad=pad,
                boundary_condition=boundary_condition,
                fill=array_padded_fill,
            )

            # # Manage input mask if any is given
            # if array_in_mask is not None:
                # mask_padded = source_extent_pad(
                    # array_src=array_in_mask,
                    # pad=pad,
                    # boundary_condition=boundary_condition,
                    # fill=mask_padded_fill,
                # )
            # elif boundary_condition is None:
                # # Mask handling:
                # # - A default mask must be created if:
                # #   * No mask is provided AND
                # #   * No boundary condition is specified
                # # - No mask creation is needed if:
                # #   * A boundary condition is provided and applied to the input array
                # #   * In this case, all points are considered valid by default
                # mask_profile = ArrayProfile(
                    # shape=(array_in_shape[1], array_in_shape[2]),
                    # ndim=2,
                    # dtype=np.uint8,
                # )
                # (mask_padded_shape, mask_source_window) = get_array_padded_shape(mask_profile, pad)

                # # Allocate buffer for padded mask and fill it with INVALID value
                # mask_padded = np.full(
                    # mask_padded_shape, Validity.INVALID, dtype=np.uint8, order="C"
                # )

                # # Mark original region as VALID
                # mask_padded[mask_source_window] = Validity.VALID

            # Mask handling:
            # 1. If a mask is provided if must be padded considering boundary condition
            # 2. Otherwise :
            #   2.1 A default mask must be created if:
            #    - No boundary condition is specified in order to mark domain extension as invalid.
            #    - A boundary condition is specified and the interpolator is a BSpline (validity 
            #      for domain extension has to be set)
            #   2.2 No mask creation is needed if:
            #     - A boundary condition is provided and interpolator is not BSpline : in this case,
            #      all points are considered valid by default
            if array_in_mask is not None:
                mask_padded = source_extent_pad(
                    array_src=array_in_mask,
                    pad=pad,
                    boundary_condition=boundary_condition,
                    fill=mask_padded_fill,
                )
            elif boundary_condition is None or is_bspline(interp):
                mask_profile = ArrayProfile(
                    shape=(array_in_shape[1], array_in_shape[2]),
                    ndim=2,
                    dtype=np.uint8,
                )
                (mask_padded_shape, mask_source_window) = get_array_padded_shape(mask_profile, pad)

                if boundary_condition is None:
                    # Allocate buffer for padded mask and fill it with INVALID value
                    mask_padded = np.full(
                        mask_padded_shape, Validity.INVALID, dtype=np.uint8, order="C"
                    )

                    # Mark original region as VALID
                    mask_padded[mask_source_window] = Validity.VALID
                else:
                    # Interpolator is BSpline
                    # Here we allocate a full valid mask.
                    # TODO - Computing Improvement : we can still have mask as
                    # None and create the appropriate mask ater bspline 
                    # prefiltering
                    mask_padded = np.full(
                        mask_padded_shape, Validity.VALID, dtype=np.uint8, order="C"
                    )

            # substitute array_in with array_padded
            array_in = array_padded

            # Update array padded shape to match the effective shape
            array_padded_shape = array_padded.shape
            if array_padded.ndim == 2:
                array_padded_shape = tuple(np.insert(array_padded_shape, 0, array_in_shape[0]))
            array_in_shape = array_padded_shape

            # subsitute array_in_mask with mask_padded
            array_in_mask = mask_padded

            # We've applied padding to `array_in` and optional associated mask,
            # so we must account for the implied shift in coordinates.
            # Note: Standalone mode is incompatible with `array_in_origin` as
            # input, but we can use it here for the coming core Rust function
            # call.
            array_in_origin = (pad[0][0], pad[1][0])

        # Note : If we want to apply low-pass filtering for antialiasing this is
        # the right place. But first we would have to integrate the required
        # margin for antialiasing into the total margins requirement.

        # Manage interpolator preprocessings
        if is_bspline(interp):
            # Prefiltering is performed in-place on the available data.
            array_bspline_prefiltering(
                array_in=array_in,  # thats the previously read buffer
                array_in_mask=array_in_mask,
                interp=interp,  # The interpolator
            )

    array_in = array_in.reshape(-1)
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
            interp=interp,
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
