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
from typing import Any, NamedTuple, NoReturn, Optional, Tuple, Union

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
from gridr.core.utils.array_window import window_indices

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

    This function computes the minimal read window from the source array
    required to resample data onto the target grid. It accounts for
    interpolation margins and handles boundary cases where the required window
    extends beyond the source array bounds.

    The calculation proceeds in three steps:

    1. Compute grid metrics from valid grid coordinates
    2. Apply interpolation margins to determine the required source extent
    3. Adjust for boundary conditions and compute necessary padding

    Warning : the padding does not guarantee that all the grid target
    coordinates lie within the final array. It only guarantee that target points
    that exist in the source image domain will have a sufficient neighborhood to
    perform interpolation.

    Parameters
    ----------
    interp : Interpolator
        Interpolator instance that defines the required margins for
        interpolation.
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
        Oversampling factor as (row_resolution, col_resolution). Value of 1
        indicates full resolution; higher values indicate coarser grids.

    grid_nodata : int or float, optional
        Value in `grid_row` and `grid_col` indicating invalid cells. Mutually
        exclusive with `grid_mask` (exclusivity enforced in core method).

    grid_mask : np.ndarray, optional
        Integer mask array for the grid. Cells matching `grid_mask_valid_value`
        are considered valid. Must have the same shape as `grid_row` and
        `grid_col`.

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
        Format: ``[[row_start, row_end], [col_start, col_end]]``. Use this
        window for raster IO operations.

    array_src_win_marged : np.ndarray, shape (2, 2)
        Desired read window with margins applied, before boundary correction.
        Format: ``[[row_start, row_end], [col_start, col_end]]``.

    pad : np.ndarray, shape (2, 2)
        Padding required if the marged window extends outside the source array.
        Format: ``[[top_pad, bottom_pad], [left_pad, right_pad]]``.

    grid_metrics : Union[PyGeometryBoundsF64, None]
        A structure containing the computed boundaries (`PyGeometryBoundsF64`)
        or `None` if no valid boundaries can be computed (e.g., empty grid).

    Notes
    -----
    - If no valid grid points are found, returns None for all outputs
    - The `safecheck_src_boundaries` option is useful to detect grid topology
      issues that could cause panic
    - Padding may be required when the grid extends beyond source boundaries,
      which should be filled using appropriate boundary conditions

    See Also
    --------
    array_compute_resampling_grid_geometries : Computes grid metrics from
    coordinates read_win_from_grid_metrics : Derives read window from grid
    metrics source_extent_pad : Applies padding to source arrays

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

    # TODO? : Add an extra margin for bspline to ensure consistency with no mask
    # all a full valid mask
    # if is_bspline(interp):
    #    margin += 1

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
    # First init returned values to None then proceed with defined grid_metrics
    array_src_win_read, array_src_win_marged, pad = None, None, None

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

    return array_src_win_read, array_src_win_marged, pad, grid_metrics


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


class ResamplingMaskStrategy(NamedTuple):
    """Result of mask strategy resolution."""

    mask_kind: Optional[str]
    """One of ``'none'``, ``'safe_region'`` or ``'binary'``."""

    safe_region: Optional[np.ndarray]
    """The safe window [[row_min, row_max], [col_min, col_max]] (inclusives
    boundaries) or None."""

    needs_mask_alloc: bool
    """Whether a mask buffer must be allocated (only when no user mask is
    provided)."""

    pad_fill: Optional[Any]

    boundary_condition: Optional[str]


def _create_safe_region(
    pad: np.ndarray,
    trust_padding: bool,
    array_in_shape: Tuple[int, ...],
) -> np.ndarray:
    """Compute the safe region window within the padded array.

    The safe region is the rectangular zone where all data is considered
    valid and no mask check is required during interpolation.

    When padding is applied:

    - ``trust_padding=True``: the safe region covers the entire padded
      array (original data + extrapolated padding).
    - ``trust_padding=False``: only the original (non-padded) data zone
      is considered safe.

    When no padding is applied, the safe region always covers the full
    array regardless of ``trust_padding``.

    Coordinates are expressed in the post-padded array reference frame.

    Parameters
    ----------
    pad : np.ndarray, shape (2, 2)
        Padding amounts ``[[top, bottom], [left, right]]``.

    trust_padding : bool
        If ``True``, the padded zone is considered valid (e.g. filled
        by a boundary condition that produces exploitable values).
        If ``False``, only the original data zone is safe.
        Ignored when no padding is applied.

    array_in_shape : tuple of int
        Shape of the **original** (pre-padded) input array, as
        ``(nvar, nrow, ncol)`` or ``(nrow, ncol)``.

    Returns
    -------
    safe_region : np.ndarray, shape (2, 2)
        Window in the padded array with inclusive boundaries:
        ``[[row_start, row_end], [col_start, col_end]]``.
    """
    pad = np.asarray(pad)
    has_pad = np.any(pad != 0)
    sr_start_row, sr_start_col = 0, 0
    sr_nrow, sr_ncol = None, None
    if len(array_in_shape) == 3:
        _, sr_nrow, sr_ncol = array_in_shape
    elif len(array_in_shape) == 2:
        sr_nrow, sr_ncol = array_in_shape

    if has_pad:
        if trust_padding:
            # The full padded array can be considered safe
            sr_nrow += max(0, pad[0][0]) + max(0, pad[0][1])
            sr_ncol += max(0, pad[1][0]) + max(0, pad[1][1])
        else:
            # Only the original pre-paddded array can be considered safe
            sr_start_row = max(0, pad[0][0])
            sr_start_col = max(0, pad[1][0])

    sr = np.array(
        [
            [sr_start_row, sr_start_row + sr_nrow - 1],
            [sr_start_col, sr_start_col + sr_ncol - 1],
        ]
    )
    return sr


def resolve_mask_strategy(
    interp: Any,
    pad: np.ndarray,
    array_in_mask: Optional[np.ndarray],
    boundary_condition: Optional[str],
    trust_padding: bool = True,
    array_in_shape: Optional[Tuple[int, ...]] = None,
    array_in_mask_safe_win: Optional[np.ndarray] = None,
) -> ResamplingMaskStrategy:
    """Determine how to prepare the input mask before passing it to the
    Rust interpolation core.

    Based on the input configuration (padding, boundary condition, mask
    content, interpolator type), this function decides whether a mask
    buffer must be allocated, how to fill the padded zone, and whether
    a safe region can be identified to optimize mask checks.

    This function must be called from both the standalone and IO code
    paths to guarantee consistent behaviour. Its output drives
    ``apply_mask_strategy`` which builds the actual mask buffer.

    Decision tree
    -------------
    ::

        mask has real invalids?
        |
        +-- yes
        |   +-- safe_win provided?
        |   |   +-- yes -> safe_region (shifted by pad)
        |   |   +-- no  -> binary
        |   |
        |   +-- pad > 0?
        |       +-- yes + BC + trust -> pad_fill=VALID, BC propagated
        |       +-- yes + otherwise  -> pad_fill=INVALID
        |       +-- no               -> pad_fill=None
        |
        +-- no (mask is None or full-valid)
            |
            +-- pad = 0?
            |   +-- bspline -> safe_region (full array, alloc mask)
            |   +-- other   -> none
            |
            +-- pad > 0
                |
                +-- BC + trust?
                |   +-- bspline -> safe_region (full padded, alloc if no mask)
                |   +-- other   -> none
                |
                +-- otherwise (not trusted)
                    +-----------> safe_region (data zone only, alloc if no mask)

    Parameters
    ----------
    interp : Interpolator
        The interpolator instance. Used to determine whether B-spline
        prefiltering applies.

    pad : np.ndarray, shape (2, 2)
        Padding amounts ``[[top, bottom], [left, right]]``.

    array_in_mask : np.ndarray or None
        Optional input mask (uint8, 2D). If provided and not
        full-valid, the strategy accounts for scattered invalids.
        A full-valid mask is treated as no mask.

    boundary_condition : str or None
        Boundary condition used to fill the padded zone (``'edge'``,
        ``'reflect'``, ``'symmetric'``, ``'wrap'``).
        ``None`` means the padded zone contains zeros (invalid data).

    trust_padding : bool, default True
        If ``True`` and a boundary condition was applied, the
        extrapolated padding data is considered numerically valid.
        When ``False``, the padded zone is treated as suspect
        regardless of the boundary condition.

    array_in_shape : tuple of int
        Shape of the **original** (pre-padded) input array, as
        ``(nvar, nrow, ncol)`` or ``(nrow, ncol)``.  Required
        whenever the strategy may be ``'safe_region'``.

    array_in_mask_safe_win : np.ndarray, shape (2, 2), optional
        Caller-provided safe region within the input mask, as
        ``[[row_start, row_end], [col_start, col_end]]`` with
        inclusive boundaries, in pre-padded coordinates.
        When provided with a mask that has real invalids, promotes
        the strategy from ``'binary'`` to ``'safe_region'``.
        The region is automatically shifted to account for padding.
        Ignored when no real invalids are present in the mask.

    Returns
    -------
    ResamplingMaskStrategy
        Named tuple with fields:

        - ``mask_kind``: ``'none'``, ``'safe_region'``, or
          ``'binary'``.
        - ``safe_region``: window array (inclusive boundaries in
          post-padded coordinates) or ``None``.
        - ``needs_mask_alloc``: whether a structural mask buffer
          must be allocated.
        - ``pad_fill``: fill value for the padded zone in the mask
          (``Validity.VALID``, ``Validity.INVALID``, or ``None``).
        - ``boundary_condition``: boundary condition to apply when
          padding the mask, or ``None``.
    """
    pad = np.asarray(pad)
    has_pad = np.any(pad != 0)
    bspline = is_bspline(interp)

    if array_in_mask_safe_win is not None:
        array_in_mask_safe_win = np.asarray(array_in_mask_safe_win)

    has_real_in_mask = array_in_mask is not None and not array_in_mask.all()

    # ------------------------------------------------------------------
    # Input mask provided
    # ------------------------------------------------------------------
    if has_real_in_mask:
        mask_kind = "binary"
        safe_region = None

        if array_in_mask_safe_win is not None:
            mask_kind = "safe_region"
            safe_region = np.copy(array_in_mask_safe_win)

            # shift the safe region definition
            if has_pad:
                safe_region[0, :] += max(0, pad[0][0])
                safe_region[1, :] += max(0, pad[1][0])

        pad_fill = None
        if has_pad:
            if trust_padding and boundary_condition is not None:
                pad_fill = Validity.VALID
            else:
                pad_fill = Validity.INVALID
        bc_valid = has_pad and trust_padding
        return ResamplingMaskStrategy(
            mask_kind=mask_kind,
            safe_region=safe_region,
            needs_mask_alloc=False,
            pad_fill=pad_fill,
            boundary_condition=boundary_condition if bc_valid else None,
        )

    # ------------------------------------------------------------------
    # No input mask from here or all valid => do not consider
    # ------------------------------------------------------------------
    if not has_pad:
        # No padding - no mask (or all valid)
        if bspline:
            # BSpline prefiltering make boundary data invalid by nature
            # Here we force mask creation
            sr = _create_safe_region(pad, True, array_in_shape)
            mask_alloc = array_in_mask is None
            return ResamplingMaskStrategy(
                mask_kind="safe_region",
                safe_region=sr,
                needs_mask_alloc=mask_alloc,
                pad_fill=Validity.INVALID if mask_alloc else None,  # outside of safe region
                boundary_condition=None,  # no pad => INVALID by convention
            )
        else:
            return ResamplingMaskStrategy(
                mask_kind="none",
                safe_region=None,
                needs_mask_alloc=False,
                pad_fill=None,
                boundary_condition=None,
            )

    # ------------------------------------------------------------------
    # has_pad is True from here - input mask is all valid
    # ------------------------------------------------------------------
    if boundary_condition is not None and trust_padding:
        if bspline:
            # Safe region = all
            sr = _create_safe_region(pad, trust_padding, array_in_shape)
            bc = boundary_condition
            if array_in_mask is None:
                bc = None
            return ResamplingMaskStrategy(
                mask_kind="safe_region",
                safe_region=sr,
                needs_mask_alloc=array_in_mask is None,
                pad_fill=Validity.VALID,  # optimal for safe region all
                boundary_condition=bc,  # only used if array_i_mask is not None
            )
        else:
            return ResamplingMaskStrategy(
                mask_kind="none",
                safe_region=None,
                needs_mask_alloc=False,
                pad_fill=None,
                boundary_condition=None,
            )
    else:
        # Padding is applied but not trusted => safe_region
        sr = _create_safe_region(pad, trust_padding, array_in_shape)
        return ResamplingMaskStrategy(
            mask_kind="safe_region",
            safe_region=sr,
            needs_mask_alloc=array_in_mask is None,
            pad_fill=Validity.INVALID,
            boundary_condition=None,
        )


def check_mask_strategy(pad, strategy):
    """Validate the internal consistency of a ``ResamplingMaskStrategy``.

    Acts as a safety guard between ``resolve_mask_strategy`` (which
    builds the strategy) and ``apply_mask_strategy`` (which executes
    it).  Catches any combination of fields that would lead to
    incorrect mask construction or silent data corruption.

    The validation enforces four groups of invariants:

    1. **mask_kind constraints** -- each kind imposes requirements on
       the other fields:

       - ``'none'``: all other fields must be neutral (``None`` /
         ``False``).
       - ``'safe_region'``: ``safe_region`` must be set;
         ``needs_mask_alloc`` and ``boundary_condition`` are mutually
         exclusive (cannot allocate a new mask *and* pad an existing
         one).
       - ``'binary'``: ``needs_mask_alloc`` must be ``False`` (the
         caller provides the mask).

    2. **safe_region coherence** -- ``safe_region`` is not ``None``
       iff ``mask_kind == 'safe_region'``.

    3. **Padding constraints** -- when no padding is applied,
       ``pad_fill`` and ``boundary_condition`` must be ``None``
       (nothing to fill).  When padding is applied and a mask is
       needed, ``pad_fill`` must be ``Validity.VALID`` or
       ``Validity.INVALID``.

    4. **Allocation constraints** -- ``needs_mask_alloc`` requires
       ``mask_kind == 'safe_region'`` and a concrete ``pad_fill``
       value.

    Parameters
    ----------
    pad : np.ndarray, shape (2, 2)
        Padding amounts ``[[top, bottom], [left, right]]``.

    strategy : ResamplingMaskStrategy
        The strategy to validate.

    Raises
    ------
    ValueError
        If any invariant is violated, with a message identifying the
        incompatible fields.

    See Also
    --------
    resolve_mask_strategy : Builds the strategy.
    apply_mask_strategy : Executes the strategy (calls this first).
    """
    pad = np.asarray(pad)
    has_pad = np.any(pad != 0)

    # ------------------------------------------------------------------
    # 1. mask_kind constraints
    # ------------------------------------------------------------------
    if strategy.mask_kind == "none":
        if strategy.pad_fill is not None:
            raise ValueError(
                f"'mask_kind'='none' is incompatible with " f"'pad_fill'={strategy.pad_fill!r}"
            )
        if strategy.boundary_condition is not None:
            raise ValueError(
                f"'mask_kind'='none' is incompatible with "
                f"'boundary_condition'={strategy.boundary_condition!r}"
            )
        if strategy.needs_mask_alloc:
            raise ValueError("'mask_kind'='none' is incompatible with " "'needs_mask_alloc'=True")
        if strategy.safe_region is not None:
            raise ValueError("'mask_kind'='none' is incompatible with " "'safe_region' != None")

    # ------------------------------------------------------------------
    # 2. safe_region coherence
    # ------------------------------------------------------------------
    if strategy.safe_region is not None and strategy.mask_kind != "safe_region":
        raise ValueError(
            f"'safe_region' != None is incompatible with " f"'mask_kind'={strategy.mask_kind!r}"
        )
    if strategy.mask_kind == "safe_region" and strategy.safe_region is None:
        raise ValueError("'mask_kind'='safe_region' requires " "'safe_region' != None")
    if strategy.mask_kind == "safe_region":
        if strategy.needs_mask_alloc and strategy.boundary_condition is not None:
            raise ValueError(
                "'mask_kind'='safe_region' with 'needs_mask_alloc'=True "
                "is incompatible with "
                f"'boundary_condition'={strategy.boundary_condition!r} "
                "(cannot allocate and pad simultaneously)"
            )

    # ------------------------------------------------------------------
    # 3. Padding constraints
    # ------------------------------------------------------------------
    if not has_pad:
        if strategy.pad_fill is not None and not strategy.needs_mask_alloc:
            raise ValueError(
                f"'pad_fill'={strategy.pad_fill!r} requires padding or " "'needs_mask_alloc'=True"
            )
        if strategy.boundary_condition is not None:
            raise ValueError(
                f"'boundary_condition'={strategy.boundary_condition!r} "
                "is incompatible with no padding"
            )
    else:
        if strategy.mask_kind != "none" and strategy.pad_fill not in (
            Validity.VALID,
            Validity.INVALID,
        ):
            raise ValueError(
                f"'pad_fill'={strategy.pad_fill!r} must be VALID or INVALID "
                f"when padding is applied and 'mask_kind'={strategy.mask_kind!r}"
            )

    # ------------------------------------------------------------------
    # 4. Allocation constraints
    # ------------------------------------------------------------------
    if strategy.needs_mask_alloc:
        if strategy.mask_kind in ("binary", "none"):
            raise ValueError(
                f"'needs_mask_alloc'=True is incompatible with "
                f"'mask_kind'={strategy.mask_kind!r}"
            )
        if strategy.pad_fill not in (Validity.VALID, Validity.INVALID):
            raise ValueError(
                f"'needs_mask_alloc'=True requires "
                f"'pad_fill' to be VALID or INVALID, got {strategy.pad_fill!r}"
            )


def apply_mask_strategy(
    array_in_mask: Optional[np.ndarray],
    pad: np.ndarray,
    array_in_shape: Tuple[int, ...],
    strategy: ResamplingMaskStrategy,
) -> Optional[np.ndarray]:
    """Build the final mask buffer ready to pass to the Rust core.

    Executes the strategy produced by ``resolve_mask_strategy``,
    performing validation, optional allocation, and optional padding.

    Behaviour per ``mask_kind``:

    - ``'none'``: returns ``None`` -- no mask is passed to Rust.
    - ``'binary'``: pads ``array_in_mask`` with ``strategy.pad_fill``
      (and ``strategy.boundary_condition`` if set) when padding is
      present; otherwise returns it as-is.
    - ``'safe_region'``:

      - ``needs_mask_alloc=True``: allocates a new buffer filled with
        ``strategy.pad_fill``, then marks the safe region as
        ``Validity.VALID``.
      - ``needs_mask_alloc=False``: pads ``array_in_mask`` with
        ``strategy.pad_fill`` when padding is present; otherwise
        returns it as-is.

    Parameters
    ----------
    array_in_mask : np.ndarray or None
        The original (pre-padded) input mask, or ``None``.  Required
        when ``strategy.mask_kind`` is not ``'none'`` and
        ``strategy.needs_mask_alloc`` is ``False``.

    pad : np.ndarray, shape (2, 2)
        Padding amounts ``[[top, bottom], [left, right]]``.

    array_in_shape : tuple of int
        Shape of the **original** (pre-padded) input array, as
        ``(nvar, nrow, ncol)`` or ``(nrow, ncol)``.  Used to compute
        the padded mask shape when ``needs_mask_alloc`` is ``True``.

    strategy : ResamplingMaskStrategy
        Strategy produced by ``resolve_mask_strategy``.  Validated by
        ``check_mask_strategy`` before execution.

    Returns
    -------
    np.ndarray or None
        The final mask buffer (uint8, 2D, C-contiguous) to pass to the
        Rust interpolation core, or ``None`` when no mask is needed.

    Raises
    ------
    ValueError
        If ``strategy`` is internally inconsistent (via
        ``check_mask_strategy``), or if a required ``array_in_mask``
        is missing.

    See Also
    --------
    resolve_mask_strategy : Builds the strategy.
    check_mask_strategy : Validates the strategy.
    """
    pad = np.asarray(pad)
    has_pad = np.any(pad != 0)
    final_mask = None

    # First check strategy consistency
    check_mask_strategy(pad, strategy)

    if strategy.mask_kind == "binary":
        if array_in_mask is None:
            raise ValueError("'strategy.mask_kind' = 'binary' requires a mask")
        final_mask = array_in_mask
        if has_pad:
            final_mask = source_extent_pad(
                array_src=array_in_mask,
                pad=pad,
                boundary_condition=strategy.boundary_condition,
                fill=strategy.pad_fill,
            )
    elif strategy.mask_kind == "safe_region":
        if strategy.needs_mask_alloc:
            mask_profile = ArrayProfile(
                shape=array_in_shape,
                ndim=len(array_in_shape),
                dtype=np.uint8,
            )
            mask_profile.make_2d()

            mask_padded_shape, _ = get_array_padded_shape(mask_profile, pad)
            # Create a full invalid mask
            final_mask = np.full(mask_padded_shape, strategy.pad_fill, dtype=np.uint8, order="C")
            # Make valid the safe region
            if strategy.pad_fill != Validity.VALID:
                final_mask[window_indices(strategy.safe_region)] = Validity.VALID

        else:
            if array_in_mask is None:
                raise ValueError("'strategy.mask_kind' = 'safe_region' must return a mask")
            # safe_region : we dont need alloc if an existing array_in_mask
            # has been passed
            # if no pad : the mask is already OK
            # if pad : apply padding
            final_mask = array_in_mask
            if has_pad:
                final_mask = source_extent_pad(
                    array_src=array_in_mask,
                    pad=pad,
                    boundary_condition=strategy.boundary_condition,
                    fill=strategy.pad_fill,
                )
    return final_mask


def standalone_preprocessing(
    interp: Any,
    array_in: np.ndarray,
    array_in_shape: Tuple[int, ...],
    array_in_origin: Optional[Tuple[float, float]],
    grid_row: np.ndarray,
    grid_col: np.ndarray,
    grid_resolution: Tuple[int, int],
    grid_nodata: Optional[float],
    grid_mask: Optional[np.ndarray],
    grid_mask_valid_value: Optional[int],
    win: Optional[np.ndarray],
    array_in_mask: Optional[np.ndarray],
    array_in_mask_safe_win: Optional[np.ndarray],
    boundary_condition: Optional[str],
    trust_padding: Optional[bool],
    check_boundaries: bool,
    logger_msg_prefix: Optional[str],
    logger: Optional[logging.Logger],
) -> Tuple[np.ndarray, Tuple[int, ...], Optional[Tuple[float, float]], Optional[np.ndarray], bool]:
    """Perform all preprocessing steps required before the Rust resampling
    call in standalone mode.

    This function is the single entry point for standalone preprocessing.
    It sequentially handles source extent computation, data padding, mask
    preparation, and B-spline prefiltering.

    Processing steps
    ----------------
    1. Validate that ``array_in_origin`` is zero (shifting is not
       supported in standalone mode).
    2. Initialize the interpolator (required for B-spline).
    3. Compute the minimal source read window and required padding via
       ``calculate_source_extent``.
    4. Pad ``array_in`` if needed, update ``array_in_shape`` and
       ``array_in_origin`` accordingly.
    5. Prepare the mask (resolve and apply the mask strategy)
    6. If the interpolator is a B-spline, run prefiltering in place on
       ``array_in`` and ``array_in_mask``.

    Parameters
    ----------
    interp : Interpolator
        Initialised interpolator instance.

    array_in : np.ndarray
        Source data array (2D or 3D, C-contiguous).

    array_in_shape : tuple of int
        Shape of ``array_in`` as ``(nvar, nrow, ncol)`` (always 3D).

    array_in_origin : tuple of float or None
        Must be ``None`` or ``(0.0, 0.0)`` in standalone mode.

    grid_row : np.ndarray
        Row coordinates of the target grid.

    grid_col : np.ndarray
        Column coordinates of the target grid.

    grid_resolution : tuple of int
        Oversampling factor ``(row_resolution, col_resolution)``.

    grid_nodata : float or None
        NoData value in the grid coordinate arrays.

    grid_mask : np.ndarray or None
        Optional validity mask for the grid.

    grid_mask_valid_value : int or None
        Value in ``grid_mask`` designating valid cells.

    win : np.ndarray or None
        Target sub-window within the full-resolution grid.

    array_in_mask : np.ndarray or None
        Optional input validity mask for ``array_in`` (uint8, 2D).

    array_in_mask_safe_win : np.ndarray or None
        Caller-provided safe region within ``array_in_mask``, in
        pre-padded coordinates.  Forwarded to ``prepare_mask``.

    boundary_condition : str or None
        Boundary condition for padding (``'edge'``, ``'reflect'``,
        ``'symmetric'``, ``'wrap'``, or ``None`` for zero-fill).

    trust_padding : bool or None
        Whether extrapolated padding values are considered valid.
        Forwarded to ``prepare_mask``.

    check_boundaries : bool
        Passed through unchanged to the return value.

    logger_msg_prefix : str or None
        Prefix for log messages.

    logger : logging.Logger or None
        Logger instance.

    Returns
    -------
    array_in : np.ndarray
        Padded (and prefiltered if B-spline) data array.

    array_in_shape : tuple of int
        Updated shape of the padded ``array_in`` as
        ``(nvar, nrow, ncol)``.

    array_in_origin : tuple of float or None
        Updated coordinate origin accounting for padding offset.

    array_in_mask : np.ndarray or None
        Final mask buffer ready to pass to the Rust core, or ``None``
        when no mask is needed.

    safe_region : np.ndarray or None
        Safe region window within the padded array
        (``[[row_start, row_end], [col_start, col_end]]``, inclusive),
        or ``None`` if unavailable.

    check_boundaries : bool
        Passed through from input unchanged.

    Raises
    ------
    ValueError
        If ``array_in_origin`` is non-zero in standalone mode.

    See Also
    --------
    resolve_mask_strategy : Builds the strategy.
    apply_mask_strategy : Executes the strategy.
    calculate_source_extent : Source window and padding computation.
    """
    # Validate array_in_origin for standalone mode
    if array_in_origin is not None and np.any(np.asarray(array_in_origin) != 0.0):
        raise ValueError("Shifting the array origin is not available for standalone mode")

    # Initialize the interpolator - required for B-spline for instance
    interp.initialize()

    # Compute required source extent in order to compute resampling from the grid
    (
        _,
        _, 
        pad,
        grid_metrics,
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

    array_in_shape_0 = array_in_shape
    if grid_metrics is None:
        raise ValueError(
            "Cannot compute grid metrics. Please check your grid !"
        )
    
    # Apply padding to data if needed
    # Note : pad can be None for a full out of domain grid
    if pad is None:
        pad = np.array([[0, 0], [0, 0]])
    if np.any(pad != 0):
        array_padded_fill = 0 if boundary_condition is None else None
        array_in = source_extent_pad(
            array_src=array_in,
            pad=pad,
            boundary_condition=boundary_condition,
            fill=array_padded_fill,
        )
        # Update array padded shape to match its real shape
        array_in_shape = array_in.shape
        if array_in.ndim == 2:
            array_in_shape = (array_in_shape_0[0],) + array_in_shape
        # Account for the implied shift in coordinates due to padding
        array_in_origin = (pad[0][0], pad[1][0])

    # Manage and prepare mask
    # Note : we must pass the original shape not the padded one
    # The method updates the safe region if required
    strategy = resolve_mask_strategy(
        interp=interp,
        pad=pad,
        boundary_condition=boundary_condition,
        array_in_mask=array_in_mask,
        trust_padding=trust_padding,
        array_in_shape=array_in_shape_0,
        array_in_mask_safe_win=array_in_mask_safe_win,
    )

    array_in_mask = apply_mask_strategy(
        array_in_mask=array_in_mask, pad=pad, array_in_shape=array_in_shape_0, strategy=strategy
    )

    check_boundaries = check_boundaries  # or strategy.check_boundaries
    safe_region = strategy.safe_region

    # Note : If we want to apply low-pass filtering for antialiasing this is
    # the right place. But first we would have to integrate the required
    # margin for antialiasing into the total margins requirement.

    # Manage interpolator preprocessings
    if is_bspline(interp):
        # Prefiltering is performed in-place on the available data.

        # TODO : pass safe-region if any
        array_bspline_prefiltering(
            array_in=array_in,
            array_in_mask=array_in_mask,
            interp=interp,
        )
        # Update safe region after prefiltering
        # NOTE : cf previous TODO => use safe-region in preprocessing
        # For now we consider that there is no safe region after preprocessing
        safe_region = None

    return (
        array_in,
        array_in_shape,
        array_in_origin,
        array_in_mask,
        safe_region,
        check_boundaries,
    )


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
    array_in_mask_safe_win: Optional[np.ndarray] = None,
    grid_mask: Optional[np.ndarray] = None,
    grid_mask_valid_value: Optional[int] = 1,
    grid_nodata: Optional[float] = None,
    array_out_mask: Optional[Union[np.ndarray, bool]] = None,
    check_boundaries: bool = True,
    interp_kwargs: Optional[dict] = None,
    standalone: Optional[bool] = True,
    boundary_condition: Optional[str] = None,
    trust_padding: Optional[bool] = True,
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

    array_in_mask_safe_win : Optional[np.ndarray], shape (2, 2), default None
        Known all-valid sub-region within ``array_in_mask``, expressed
        as ``[[row_start, row_end], [col_start, col_end]]`` with
        inclusive boundaries in the **original** (pre-padded) array
        coordinate system.  When provided, promotes the mask strategy
        from ``'binary'`` to ``'safe_region'``, allowing the Rust core
        to skip per-pixel mask checks for stencils that fall entirely
        within this zone.  Only meaningful when ``array_in_mask`` is
        also provided and has real invalids.  Ignored in integrated mode
        (``standalone=False``).

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
        - `None`: Zero padding is applied. If insufficient data is available for
          interpolation, those regions will be marked as invalid in the mask.

        The boundary condition applies to ``array_in``. For the mask, the
        behaviour depends on ``trust_padding``:

        - If ``trust_padding=True`` and a boundary condition is set, the
          padded mask zone is marked as ``Validity.VALID`` — the
          extrapolated data is considered exploitable.
        - If ``trust_padding=False`` or ``boundary_condition=None``, the
          padded mask zone is marked as ``Validity.INVALID`` regardless of
          how ``array_in`` was padded.

    When ``array_in_mask`` is provided and contains real invalids, it
    is padded using the same boundary condition as ``array_in``, with
    the fill value determined by ``trust_padding`` as above.

    trust_padding : bool, default True
        Controls how the padded zone is marked in the mask when padding
        is applied in standalone mode.  If ``True`` and a
        ``boundary_condition`` is set, the padded zone is considered
        valid (``Validity.VALID``) and no mask check is enforced there.
        If ``False``, the padded zone is always marked as
        ``Validity.INVALID`` regardless of the boundary condition,
        ensuring conservative behaviour.
        Ignored when ``standalone=False``.

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

    # Get consistent 3d shape for input array
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

    # Execute standalone preprocessing if needed
    if standalone:
        (
            array_in,
            array_in_shape,
            array_in_origin,
            array_in_mask,
            array_in_mask_safe_region,
            check_boundaries,
        ) = standalone_preprocessing(
            interp=interp,
            array_in=array_in,
            array_in_shape=array_in_shape,
            array_in_origin=array_in_origin,
            grid_row=grid_row,
            grid_col=grid_col,
            grid_resolution=grid_resolution,
            grid_nodata=grid_nodata,
            grid_mask=grid_mask,
            grid_mask_valid_value=grid_mask_valid_value,
            win=win,
            array_in_mask=array_in_mask,
            array_in_mask_safe_win=array_in_mask_safe_win,
            boundary_condition=boundary_condition,
            trust_padding=trust_padding,
            check_boundaries=check_boundaries,
            logger_msg_prefix=logger_msg_prefix,
            logger=logger,
        )

    # Flatten data to pass to Rust function
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

    # TODO : USE array_in_mask_safe_win

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
