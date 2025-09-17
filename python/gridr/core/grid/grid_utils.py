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
Grid utils module
"""
from typing import Optional, Tuple, Union

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from gridr.cdylib import (
    PyArrayWindow2,
    PyGridGeometriesMetricsF64,
    py_array1_compute_resampling_grid_geometries_f64_f64,
)
from gridr.core.utils.array_utils import ArrayProfile
from gridr.core.utils.array_window import (
    window_apply,
    window_check,
    window_expand_ndim,
    window_shape,
)

F64_F64 = (np.dtype("float64"), np.dtype("float64"))

# The first element in the tuple key represents the type used for
# PyGridGeometriesMetricsF64 members. Only float64 is considered by now.
PY_ARRAY_COMPUTE_RESAMPLING_GRID_GEOMETRIES_FUNC = {
    F64_F64: py_array1_compute_resampling_grid_geometries_f64_f64,
}


def array_compute_resampling_grid_geometries(
    grid_row: np.ndarray,
    grid_col: np.ndarray,
    grid_resolution: Tuple[int, int],
    win: Optional[np.ndarray] = None,
    grid_mask: Optional[np.ndarray] = None,
    grid_mask_valid_value: Optional[int] = 1,
    grid_nodata: Optional[float] = None,
) -> Union[PyGridGeometriesMetricsF64, None]:
    """Computes resampling grid geometries metrics from given row and column
    grids.

    This function analyzes the validity of grid points (nodes) along rows and
    columns to determine bounding boxes and a transition matrix for resampling
    operations on grids. It returns detailed information about the source and
    destination grid bounds, as well as the transition matrix describing the
    linear transformation between source and destination grids.

    This method wraps a Rust function
    (`py_array1_compute_resampling_grid_geometries_f64_f64`) for grid metrics
    computation.

    Parameters
    ----------
    grid_row : np.ndarray
        A 2D array representing the row coordinates of the target grid,
        with the same shape as `grid_col`. The coordinates target row
        positions in the `array_in` input array.

    grid_col : np.ndarray
        A 2D array representing the column coordinates of the target grid,
        with the same shape as `grid_row`. The coordinates target column
        positions in the `array_in` input array.

    grid_resolution : Tuple[int, int]
        A tuple specifying the oversampling factor for the grid along rows and
        columns. A resolution value of 1 represents full resolution, while
        higher values indicate lower resolution grids.

    win : Optional[np.ndarray], default None
        An optional window (or sub-region) of the grid to limit the
        computation to a specific target region. The window is defined as a
        list of tuples containing the first and last indices for each dimension.
        If `None`, the entire grid is processed.

    grid_mask : Optional[np.ndarray], default None
        An optional integer mask array for the grid. Grid cells
        corresponding to `grid_mask_valid_value` are considered **valid**;
        all other values indicate **invalid** cells and will result in
        `nodata_out` in the output array. If not provided, the entire grid
        is considered valid. The grid mask must have the same shape as
        `grid_row` and `grid_col`.

    grid_mask_valid_value : Optional[int], default 1
        The value in `grid_mask` that designates a **valid** grid cell.
        All values in `grid_mask` that differ from this will be treated as
        **invalid**. This parameter is required if `grid_mask` is provided.

    grid_nodata : Optional[float], default None
        The value in `grid_row` and `grid_col` to consider as **invalid**
        cells. Note that this option is exclusive with `grid_mask`. This
        exclusivity is managed within the core bound method.

    Returns
    -------
    Union[PyGridGeometriesMetricsF64, None]
        A structure containing the computed metrics (`PyGridGeometriesMetricsF64`)
        or `None` if no valid metrics can be computed (e.g., empty grid).

    Raises
    ------
    Exception
        If the underlying Rust function
        `py_array1_compute_resampling_grid_geometries_f64_*` is not available
        for the provided input types.

    Notes
    -----

    - This method serves as a preprocessing step prior to resampling raster-like
      data onto a target coordinate grid. The resulting transition matrix
      facilitates effective anti-aliasing management.
    - The computed source bounds are intended to restrict the read window of the
      input raster, optimizing data access. Similarly, the destination bounds
      define the output grid window, preventing  unnecessary processing of
      nodata regions.
    - The method is designed to support tiled processing workflows by accepting
      an optional window parameter, enabling integration within tile-based
      operations.

    Limitations
    -----------

    - The method assumes that all input arrays (`grid_row`, `grid_col`, etc.)
      are C-contiguous. If any of them are not, the method may raise an
      assertion error.
    - The method assumes that the grid-related arrays (`grid_row`, `grid_col`,
      `grid_mask`) have the same shapes. Mismatched shapes will raise an
      assertion error.
    - The `win` parameter, if provided, must be compatible with the grid shape.
      If `win` exceeds the bounds of the grid, an error may occur.
    - The method does not handle invalid or missing values in the input arrays
      or masks beyond what's specified by `grid_mask` or `grid_nodata`. Users
      are responsible for ensuring any invalid or missing data is appropriately
      handled before calling this method.

    """
    ret = None
    assert grid_row.flags.c_contiguous is True
    assert grid_col.flags.c_contiguous is True

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

    func_types = (np.dtype("float64"), grid_row.dtype)

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
        func = PY_ARRAY_COMPUTE_RESAMPLING_GRID_GEOMETRIES_FUNC[func_types]
    except KeyError as err:
        raise Exception(
            "py_array1_compute_resampling_grid_geometries_ function"
            f" not available for types {func_types}"
        ) from err
    else:
        ret = func(
            grid_row=grid_row,
            grid_col=grid_col,
            grid_shape=grid_shape,
            grid_resolution=grid_resolution,
            grid_mask=grid_mask,
            grid_mask_valid_value=grid_mask_valid_value,
            grid_nodata=grid_nodata,
            grid_win=py_grid_win,
        )
    return ret


def interpolate_grid(
    grid: Optional[np.ndarray],
    grid_mask: Optional[np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    x_new: np.ndarray,
    y_new: np.ndarray,
    dtype: Optional[np.dtype] = None,
    mask_binarize_precision: float = 1e-6,
    mask_dtype: np.dtype = np.uint8,
) -> Tuple[np.ndarray]:
    """Interpolate a 3-dimensional grid and its associated 2-dimensional mask.

    The first dimension of the grid contains the variable. This function
    performs linear interpolation of both the grid data and its mask onto a new
    coordinate mesh defined by `x_new` and `y_new`.

    We assume here that the input mask follows the convention where all non-zero
    values are considered masked. The output mask will be binarized (i.e.,
    values equal to 0 or 1) according to a binarization threshold applied to the
    interpolated mask values:

    .. math::
        final\\_mask(i,j) = 0 \\quad \\text{only if} \\quad |interp\\_mask(i,j)| < threshold

    The grid is interpolated on the mesh generated by the `x_new` and `y_new`
    coordinates, using a linear interpolation method.

    Parameters
    ----------
    grid : Optional[np.ndarray]
        Input 3-dimensional grid. Its first dimension should represent different
        variables or bands, while the subsequent two dimensions correspond to
        spatial (row, column) data.

    grid_mask : Optional[np.ndarray]
        Input 2-dimensional mask. This mask should have the same spatial
        dimensions as `grid` (i.e., its last two dimensions). Non-zero values
        are treated as masked.

    x : np.ndarray
        1D coordinates associated with the input `grid` (and `grid_mask`) for
        column indexes.

    y : np.ndarray
        1D coordinates associated with the input `grid` (and `grid_mask`) for
        row indexes.

    x_new : np.ndarray
        1D coordinates associated with the interpolated grid and mask for
        columns. This defines the new horizontal sampling of the output grid.

    y_new : np.ndarray
        1D coordinates associated with the interpolated grid and mask for rows.
        This defines the new vertical sampling of the output grid.

    dtype : Optional[np.dtype], default None
        Data type to use for computation and for the output interpolated grid.
        If `None`, it uses the same dtype as the input `grid`.

    mask_binarize_precision : float, default 1e-6
        Threshold used for the interpolated mask's binarization. Values with an
        absolute magnitude less than this threshold will be set to 0 in the
        output mask.

    mask_dtype : np.dtype, default np.uint8
        Data type of the output mask. This should typically be `np.uint8`
        (for 0 or 1 values) or `bool`.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:

        -   The interpolated 3-dimensional grid.
        -   The binarized 2-dimensional interpolated mask.

    """
    interp_grid = None
    interp_grid_mask = None

    # Create the "sparse" coordinates grid in order to preserve memory
    x_new_sparse, y_new_sparse = np.meshgrid(x_new, y_new, indexing="xy", sparse=True)

    if dtype is None:
        dtype = grid.dtype

    # Init the output grid
    if grid is not None:
        n_vars = grid.shape[0]
        interp_grid = np.empty((grid.shape[0], len(y_new), len(x_new)), dtype=dtype)

        # Loop on each variable to perform interpolation
        for i in range(n_vars):
            # Create the interpolator
            interpolator = RegularGridInterpolator(
                (y, x), grid[i, :, :], method="linear", bounds_error=False, fill_value=np.nan
            )
            # Perform the interpolation
            interp_grid[i, :, :] = interpolator((y_new_sparse, x_new_sparse))

    # Perform interpolation on mask
    if grid_mask is not None:
        # Init the mask
        interp_grid_mask = np.empty(grid_mask.shape)

        interpolator = RegularGridInterpolator(
            (y, x), grid_mask[:, :], method="linear", bounds_error=False, fill_value=np.nan
        )
        # Perform the interpolation
        interp_grid_mask = interpolator((y_new_sparse, x_new_sparse))
        # The interpolator will generate interpolated values
        # If we are strict we will only consider unmasked data to have
        # strictly the mask_value (almost equal with a precision)
        interp_grid_mask = (np.abs(interp_grid_mask) >= mask_binarize_precision).astype(mask_dtype)

    return interp_grid, interp_grid_mask


def oversample_regular_grid(
    grid: Optional[np.ndarray],
    grid_oversampling_row: int,
    grid_oversampling_col: int,
    grid_mask: Optional[np.ndarray],
    dtype: Optional[np.dtype] = None,
    grid_mask_binarize_precision: Optional[float] = 1e-6,
    grid_mask_dtype: np.dtype = np.uint8,
    win: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Get a linearly interpolated oversampled grid from the input grid.

    This function takes an input grid and optionally an associated mask, then
    oversamples them to a higher resolution based on provided oversampling
    factors. It handles the definition of the target interpolation coordinates
    and leverages `interpolate_grid` for the actual resampling. The output grid
    and mask will have dimensions scaled by `grid_oversampling_row` and
    `grid_oversampling_col`.

    Parameters
    ----------
    grid : Optional[np.ndarray]
        Input 3-dimensional grid. Its first dimension represents different
        variables or bands, while the subsequent two dimensions correspond
        to spatial (row, column) data. Can be `None` if only `grid_mask`
        is to be oversampled.

    grid_oversampling_row : int
        Integer factor by which to oversample the grid along the row dimension.
        A value of 1 means no oversampling in this dimension.

    grid_oversampling_col : int
        Integer factor by which to oversample the grid along the column
        dimension. A value of 1 means no oversampling in this dimension.

    grid_mask : Optional[np.ndarray]
        Optional 2-dimensional mask associated with the input grid. If `grid` is
        `None`, this mask's shape is used to determine the original grid
        dimensions.

    dtype : Optional[np.dtype], default None
        Data type to use for computation and for the output interpolated grid.
        If `None`, it attempts to infer from `grid.dtype` if `grid` is provided
        and floating-point; otherwise, a `ValueError` is raised.

    grid_mask_binarize_precision : Optional[float], default 1e-6
        Threshold used for the interpolated mask's binarization. This
        parameter is passed directly to `interpolate_grid`.

    grid_mask_dtype : np.dtype, default np.uint8
        Data type of the output mask (e.g., `np.uint8` for 0 or 1 values,
        or `bool`). This parameter is passed directly to `interpolate_grid`.

    win : Optional[np.ndarray], default None
        An optional window (or sub-region) of the *oversampled* grid to compute.
        If `None`, the entire oversampled grid is processed. The window should
        be defined in terms of the *oversampled* coordinates
        (e.g., ``((row_start, row_end), (col_start, col_end))``).

    Returns
    -------
    Tuple[Optional[np.ndarray], Optional[np.ndarray]]
        A tuple containing:

        -   `out_grid` (`Optional[np.ndarray]`): The oversampled 3-dimensional
            grid, or `None` if the input `grid` was `None`.
        -   `out_grid_mask` (`Optional[np.ndarray]`): The oversampled and
            binarized 2-dimensional mask, or `None` if the input `grid_mask` was
            `None`.

    Raises
    ------
    AssertionError
        If `grid` is provided but its number of dimensions (`ndim`) is not 3.

    AssertionError
        If the calculated or provided `win` is outside the domain of the
        oversampled grid (checked by an internal `window_check` function).

    ValueError
        If `dtype` is `None` and the function cannot infer a floating-point
        type from the input `grid`.

    """
    # Check that the grid dimension is 3
    out_grid, out_grid_mask = None, None
    nrows, ncols = None, None

    if grid is not None:
        assert grid.ndim == 3
        nrows, ncols = grid.shape[1:]
    else:
        nrows, ncols = grid_mask.shape

    # Define the window to the full data if not given
    if win is None:
        # Define the window on full grid
        # - first idx is 0
        # - last idx equals to (shape(axis) - 1) * oversampling
        # => the number of points along an axis is given by :
        #    (shape(axis) -1) * oversampling + 1
        win = np.asarray(
            [[0, (nrows - 1) * grid_oversampling_row], [0, (ncols - 1) * grid_oversampling_col]]
        )

    # Check the window is OK - for that we pass an ArrayProfile in order to
    # mock the output array's profile
    out_array_profile = ArrayProfile(
        shape=((nrows - 1) * grid_oversampling_row + 1, (ncols - 1) * grid_oversampling_col + 1),
        ndim=2,
        dtype=float,
    )

    if not window_check(out_array_profile, win):
        raise Exception("window outside of output grid domain")

    # Check dtype to use
    if dtype is None:
        if grid is not None and np.issubdtype(grid.dtype, np.floating):
            dtype = grid.dtype
        else:
            raise ValueError(
                "You must precise argument 'dtype'. Cannot tell "
                "if float32 or float64 should be used"
            )

    # Create associated coordinates
    # Input grid coordinates
    # y = np.arange(0, nrows, dtype=xy_dtype) * grid_oversampling_row
    # x = np.arange(0, ncols, dtype=xy_dtype) * grid_oversampling_col
    # Target grid coordinates
    # y_new = np.arange(win[0,0], win[0,1]+1, dtype=xy_dtype)
    # x_new = np.arange(win[1,0], win[1,1]+1, dtype=xy_dtype)

    y = np.linspace(0, (nrows - 1) * grid_oversampling_row, nrows, dtype=dtype)
    x = np.linspace(0, (ncols - 1) * grid_oversampling_col, ncols, dtype=dtype)

    y_new = np.linspace(win[0, 0], win[0, 1], win[0, 1] - win[0, 0] + 1, dtype=dtype)
    x_new = np.linspace(win[1, 0], win[1, 1], win[1, 1] - win[1, 0] + 1, dtype=dtype)

    # Interpolate the grid
    out_grid, out_grid_mask = interpolate_grid(
        grid=grid,
        grid_mask=grid_mask,
        x=x,
        y=y,
        x_new=x_new,
        y_new=y_new,
        dtype=dtype,
        mask_binarize_precision=grid_mask_binarize_precision,
        mask_dtype=grid_mask_dtype,
    )
    return out_grid, out_grid_mask


def build_grid(
    resolution: Tuple[int, int],
    grid: np.ndarray,
    grid_target_win: np.ndarray,
    grid_resolution: Tuple[int, int],
    out: np.ndarray,
    computation_dtype: Optional[np.dtype] = None,
) -> Optional[np.ndarray]:
    """Create the target resolution grid.

    This method generates a grid at a specified target resolution by resampling
    an input raster-like grid. It performs no I/O operations.
    The function supports providing a preallocated output buffer for efficiency.

    A preallocated output buffer can be passed to the method via the `out`
    argument. If provided, its shape must be consistent with the expected output
    shape determined by `grid_target_win` and `resolution`.

    The data type used for interpolation can also be specified with
    `computation_dtype`. Note that if `computation_dtype` differs from the
    output buffer's data type (or the input grid's data type if `out` is not
    provided), an implicit cast to the output data type will be performed during
    the final assignment.

    Parameters
    ----------
    resolution : Tuple[int, int]
        The desired resolution of the output grid. Currently, only full
        resolution (i.e., `(1, 1)`) is implemented. This parameter influences
        the resampling (oversampling) of the input grid.

    grid : np.ndarray
        The input grid, expected to be a 3-dimensional NumPy array. Its first
        dimension typically represents different bands or variables.

    grid_target_win : np.ndarray
        The production window for the output grid. It's provided as a
        2-dimensional array of shape (2, 2) defining
        ``((first_row, last_row), (first_col, last_col))``. This window is
        expressed in the output grid's coordinate system.

    grid_resolution : Tuple[int, int]
        The resolution (oversampling factors) of the input `grid` in row and
        column directions. For example, `(2, 2)` means the input grid is twice
        oversampled in both dimensions compared to its intrinsic resolution.

    out : np.ndarray
        An optional preallocated NumPy array buffer to store the result.
        If provided, its shape must perfectly match the expected output shape
        derived from `grid_target_win` and `resolution`.

    computation_dtype : Optional[np.dtype], default None
        An optional data type to use for the interpolation computations.
        If `None`, it defaults to the `out` array's data type if `out` is given,
        or the `grid` data type otherwise.

    Returns
    -------
    Optional[np.ndarray]
        The computed grid as a NumPy array if `out` was `None`. If `out` was
        provided, the result is written directly into it, and this function
        returns `None`.

    Raises
    ------
    ValueError
        If `resolution` or `grid` are `None`.

    ValueError
        If the input `grid` does not have 3 dimensions.

    ValueError
        If the desired output `resolution` is not `(1, 1)` (due to current
        implementation limitations).

    ValueError
        If `grid_target_win` is not a 2D window.

    ValueError
        If `grid_target_win` is outside the bounds of the full-resolution
        input grid.

    ValueError
        If `out` is provided but its shape does not match the expected
        output shape.
    """
    ret = None
    # -- Perform some checks on arguments and init optional arguments
    if resolution is None:
        raise ValueError("You must provide both the 'shape' and 'resolution' " "arguments")
    if grid is None:
        raise ValueError("You must provide the 'grid' argument")
    if grid.ndim != 3:
        raise ValueError("Input grid must have 3 dimensions")

    if ~np.all(resolution == (1, 1)):
        raise ValueError(
            "Output resolution different from full resolution have" " not been implemented yet"
        )

    grid_full_res_profile = ArrayProfile(
        shape=(
            grid.shape[0],
            (grid.shape[1] - 1) * grid_resolution[0] + 1,
            (grid.shape[2] - 1) * grid_resolution[1] + 1,
        ),
        ndim=grid.ndim,
        dtype=grid.dtype,
    )

    if grid_target_win is None:
        # Compute full size
        grid_target_win = np.asarray(
            [[0, grid_full_res_profile.shape[1] - 1], [0, grid_full_res_profile.shape[2] - 1]]
        )
    else:
        grid_target_win = np.asarray(grid_target_win)
        if grid_target_win.ndim != 2:
            raise ValueError("The argument 'grid_target_win' must be a 2d " "window")
    grid_target_win3 = window_expand_ndim(grid_target_win, (0, grid.shape[0] - 1))

    # check that the target window lies in the full resolution grid
    if not window_check(arr=grid_full_res_profile, win=grid_target_win3, axes=None):
        raise ValueError(
            "Target window error is not contained in input grid : "
            f"\n\t Input grid : {grid_full_res_profile.shape}"
            f"\n\t Window : {grid_target_win3}"
        )

    # Compute shape
    shape3 = window_shape(grid_target_win3)

    # Init output buffer if not given
    if out is None:
        out = np.zeros(shape3, dtype=grid.dtype)
        ret = out
    elif ~np.all(out.shape == shape3):
        raise ValueError("The values of the 2 arguments 'out' and 'shape' does " "not match.")

    if computation_dtype is None:
        computation_dtype = out.dtype
    # elif dtype != out.dtype:
    # raise ValueError(f"Ouput data type {out.dtype} does not match with the "
    #        f"input argument 'dtype' {dtype}")
    # -- End of argument's checks

    if grid_resolution[0] != 1 or grid_resolution[1] != 1:
        # We have to oversample the grid to the output resolution
        # FUTURE_WARNING : if resolution_out != (1,1) we will have to
        # reimplement this part
        out[:, :, :], _ = oversample_regular_grid(
            grid=grid,
            grid_oversampling_row=grid_resolution[0],
            grid_oversampling_col=grid_resolution[1],
            grid_mask=None,
            win=grid_target_win,  # the method takes a 2d window
            dtype=computation_dtype,
        )
    else:
        # No computation to perform - just select the target window
        # FUTURE_WARNING : if resolution_out != (1,1) we will have to
        # reimplement this part
        # Please note the method takes a window with same ndim as grid
        out[:, :, :] = window_apply(arr=grid, win=grid_target_win3)

    return ret
