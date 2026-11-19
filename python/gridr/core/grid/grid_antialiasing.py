#
# Copyright (c) 2024-2026 Centre National d'Etudes Spatiales (CNES).
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
Antialiasing filter of a grid resampling.

This module reads on the grid what an antialiasing filter needs, and resolves
a filter specification (see :mod:`gridr.core.filter.antialiasing`):

- :func:`estimate_transition_matrix` estimates the transition matrix ``W`` of
  the grid, in two modes, and :func:`fit_affine_least_squares` measures how
  far the grid is from an affine model;
- :func:`resolve_antialiasing_filter` estimates ``W`` once on the whole grid
  and passes it to the ``design`` method of the specification. It returns a
  :class:`~gridr.core.filter.antialiasing.KernelFilter`, or None when no
  filter is needed, with :class:`GridAntialiasingDiagnostics`.

The former :func:`compute_antialiasing_filter_from_grid` is deprecated; it is
kept for one version as a wrapper around :func:`resolve_antialiasing_filter`.
"""

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from gridr.core.filter.antialiasing import AntialiasingFilter, KernelFilter
from gridr.core.filter.antialiasing.reciprocal_cell import (
    ReciprocalCellFilter,
    tanh_gate,
)
from gridr.core.grid.grid_commons import (
    grid_regular_coords_2d,
    grid_resolution_window_safe,
)
from gridr.core.grid.grid_utils import array_compute_resampling_grid_geometries

#: Tolerance on ``grid_nodata``, the one of the grid geometry analysis
#: (``F64_TOLERANCE`` of the Rust crate): a node is invalid when its row
#: coordinate is within this distance of ``grid_nodata``.
GRID_NODATA_TOLERANCE = 1e-5


# --------------------------------------------------------------------------- #
# Grid nodes taken into account
# --------------------------------------------------------------------------- #
def _check_node_validity_args(
    grid_mask: np.ndarray | None,
    grid_mask_valid_value: int | None,
    grid_nodata: float | None,
) -> None:
    """
    Check the arguments that mark the invalid grid nodes.

    The rules are those of the grid geometry analysis: ``grid_mask`` and
    ``grid_nodata`` are mutually exclusive, and ``grid_mask_valid_value`` is
    required with ``grid_mask``.

    Parameters
    ----------
    grid_mask : np.ndarray or None
        Integer mask of the grid nodes.

    grid_mask_valid_value : int or None
        Value of ``grid_mask`` designating the valid nodes.

    grid_nodata : float or None
        Row coordinate value designating the invalid nodes.

    Raises
    ------
    ValueError
        If both ``grid_mask`` and ``grid_nodata`` are given, or if
        ``grid_mask`` is given without ``grid_mask_valid_value``.
    """
    if grid_mask is not None and grid_nodata is not None:
        raise ValueError(
            "Only one of `grid_mask` or `grid_nodata` may be provided, not both."
        )
    if grid_mask is not None and grid_mask_valid_value is None:
        raise ValueError(
            "The argument `grid_mask_valid_value` is mandatory when using `grid_mask`"
        )


def _node_window(
    grid_shape: tuple[int, int],
    grid_resolution: tuple[int, int],
    win: np.ndarray | None,
) -> np.ndarray | None:
    """
    Window of grid nodes covering a window of the full resolution grid.

    Parameters
    ----------
    grid_shape : tuple[int, int]
        Number of grid nodes ``(rows, columns)``.

    grid_resolution : tuple[int, int]
        Oversampling factors of the grid ``(rows, columns)``.

    win : np.ndarray or None
        Window of the full resolution grid, ``[[first_row, last_row],
        [first_col, last_col]]`` with inclusive bounds.

    Returns
    -------
    np.ndarray or None
        The window of grid nodes, inclusive bounds, as computed by
        :func:`~gridr.core.grid.grid_commons.grid_resolution_window_safe`;
        None when ``win`` is None.
    """
    if win is None:
        return None
    node_win, _ = grid_resolution_window_safe(
        resolution=grid_resolution, win=np.asarray(win), grid_shape=grid_shape
    )
    return node_win


def _valid_nodes(
    grid_row: np.ndarray,
    grid_resolution: tuple[int, int],
    win: np.ndarray | None,
    grid_mask: np.ndarray | None,
    grid_mask_valid_value: int | None,
    grid_nodata: float | None,
) -> np.ndarray:
    """
    Boolean mask of the grid nodes taken into account.

    A node is kept if it lies in the window of nodes covering ``win``, if its
    ``grid_mask`` value is ``grid_mask_valid_value``, and if its row
    coordinate differs from ``grid_nodata`` by at least
    :data:`GRID_NODATA_TOLERANCE`, as in the grid geometry analysis.

    Parameters
    ----------
    grid_row : np.ndarray
        Source row coordinates at the grid nodes.

    grid_resolution : tuple[int, int]
        Oversampling factors of the grid ``(rows, columns)``.

    win : np.ndarray or None
        Window of the full resolution grid, inclusive bounds.

    grid_mask : np.ndarray or None
        Integer mask of the grid nodes.

    grid_mask_valid_value : int or None
        Value of ``grid_mask`` designating the valid nodes.

    grid_nodata : float or None
        Row coordinate value designating the invalid nodes.

    Returns
    -------
    np.ndarray
        Boolean mask of the shape of the grid.

    Raises
    ------
    ValueError
        If the validity arguments are inconsistent (see
        :func:`_check_node_validity_args`).
    """
    _check_node_validity_args(grid_mask, grid_mask_valid_value, grid_nodata)
    valid = np.ones(grid_row.shape, dtype=bool)
    if grid_mask is not None:
        valid &= np.asarray(grid_mask) == grid_mask_valid_value
    if grid_nodata is not None:
        valid &= np.abs(grid_row - grid_nodata) >= GRID_NODATA_TOLERANCE
    node_win = _node_window(grid_row.shape, grid_resolution, win)
    if node_win is not None:
        in_win = np.zeros(grid_row.shape, dtype=bool)
        in_win[
            node_win[0, 0] : node_win[0, 1] + 1, node_win[1, 0] : node_win[1, 1] + 1
        ] = True
        valid &= in_win
    return valid


# --------------------------------------------------------------------------- #
# Transition matrix of the grid
# --------------------------------------------------------------------------- #
def fit_affine_least_squares(
    grid_row: np.ndarray,
    grid_col: np.ndarray,
    grid_resolution: tuple[int, int] = (1, 1),
    win: np.ndarray | None = None,
    grid_mask: np.ndarray | None = None,
    grid_mask_valid_value: int | None = 1,
    grid_nodata: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Fit an affine model to a backward grid by least squares.

    The model is ``(x, y) = p0 + W (j', i')``, where ``(i', j')`` are full
    resolution destination indices. The grid node ``(r, c)`` sits at
    ``(i', j') = (r * grid_resolution[0], c * grid_resolution[1])``.

    Parameters
    ----------
    grid_row, grid_col : np.ndarray
        Source row (``y``) and column (``x``) coordinates at the grid nodes.

    grid_resolution : tuple[int, int], optional
        Oversampling factors of the grid ``(rows, columns)``.

    win : np.ndarray or None, optional
        Window of the full resolution grid, ``[[first_row, last_row],
        [first_col, last_col]]`` with inclusive bounds. Only the grid nodes
        needed to cover it are used. None uses the whole grid.

    grid_mask : np.ndarray or None, optional
        Integer mask of the grid nodes. Nodes whose value differs from
        ``grid_mask_valid_value`` are left out. Exclusive with
        ``grid_nodata``.

    grid_mask_valid_value : int or None, optional
        Value of ``grid_mask`` designating the valid nodes.

    grid_nodata : float or None, optional
        Row coordinate value designating the invalid nodes. Exclusive with
        ``grid_mask``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        ``(W, p0, residual)``: the 2x2 matrix ``[w1 w2]``, the offset
        ``(x0, y0)`` and the largest distance between a valid node and the
        model, in source pixels.

    Raises
    ------
    ValueError
        If the validity arguments are inconsistent, or if the valid nodes are
        fewer than three or collinear: the model is then undefined.
    """
    ok = _valid_nodes(
        grid_row, grid_resolution, win, grid_mask, grid_mask_valid_value, grid_nodata
    )
    if ok.sum() < 3:
        raise ValueError("At least three valid grid nodes are needed")
    jp, ip = grid_regular_coords_2d(grid_row.shape, (0.0, 0.0), grid_resolution)
    a = np.column_stack((jp[ok], ip[ok], np.ones(ok.sum())))
    xy = np.column_stack((grid_col[ok], grid_row[ok]))
    c, _, rank, _ = np.linalg.lstsq(a, xy, rcond=None)
    if rank < 3:
        raise ValueError("The valid grid nodes are collinear: W is undefined")
    w = c[:2].T
    residual = np.linalg.norm(a @ c - xy, axis=1)
    return w, c[2], float(residual.max())


def _longest_distance_matrix(
    grid_row: np.ndarray,
    grid_col: np.ndarray,
    grid_resolution: tuple[int, int],
    win: np.ndarray | None,
    grid_mask: np.ndarray | None,
    grid_mask_valid_value: int | None,
    grid_nodata: float | None,
) -> np.ndarray:
    """
    Transition matrix computed by the grid geometry analysis.

    In every grid row, the first and the last valid nodes are found; the row
    where they are farthest apart in index gives ``w1``, their coordinate
    difference divided by the index distance times the column resolution.
    The column of longest valid span gives ``w2`` in the same way. This is the
    matrix returned by :func:`array_compute_resampling_grid_geometries`, called
    with the window of nodes covering ``win``, as the resampling does.

    Parameters
    ----------
    grid_row, grid_col : np.ndarray
        Source row and column coordinates at the grid nodes.

    grid_resolution : tuple[int, int]
        Oversampling factors of the grid ``(rows, columns)``.

    win : np.ndarray or None
        Window of the full resolution grid, inclusive bounds.

    grid_mask : np.ndarray or None
        Integer mask of the grid nodes, of type ``uint8``.

    grid_mask_valid_value : int or None
        Value of ``grid_mask`` designating the valid nodes.

    grid_nodata : float or None
        Row coordinate value designating the invalid nodes.

    Returns
    -------
    np.ndarray
        The 2x2 matrix ``[w1 w2]``.

    Raises
    ------
    ValueError
        If the grid geometry analysis finds no valid row or column, or a
        single one: ``W`` is then undefined.
    """
    metrics = array_compute_resampling_grid_geometries(
        grid_row=np.ascontiguousarray(grid_row, dtype=np.float64),
        grid_col=np.ascontiguousarray(grid_col, dtype=np.float64),
        grid_resolution=tuple(grid_resolution),
        win=_node_window(grid_row.shape, grid_resolution, win),
        grid_mask=None if grid_mask is None else np.ascontiguousarray(grid_mask),
        grid_mask_valid_value=grid_mask_valid_value,
        grid_nodata=grid_nodata,
    )
    if metrics is None:
        raise ValueError("The grid geometry analysis found no valid row or column")
    w = metrics.transition_matrix.w()
    if w is None:
        raise ValueError("The grid has a single valid row or column: W is undefined")
    return np.asarray(w, dtype=np.float64).reshape(2, 2)


def estimate_transition_matrix(
    grid_row: np.ndarray,
    grid_col: np.ndarray,
    grid_resolution: tuple[int, int] = (1, 1),
    win: np.ndarray | None = None,
    grid_mask: np.ndarray | None = None,
    grid_mask_valid_value: int | None = 1,
    grid_nodata: float | None = None,
    mode: str = "longest-distance",
) -> np.ndarray:
    """
    Estimate the transition matrix ``W = [w1 w2]`` of a backward grid.

    Both modes are exact on affine grids.

    - ``"longest-distance"`` uses, for each vector, two nodes along the row
      or the column of longest valid span. It is the matrix of the grid
      geometry analysis.
    - ``"least-square-fit"`` fits an affine model on every valid node. It
      averages a slowly varying Jacobian and filters the noise of the nodes.

    Parameters
    ----------
    grid_row, grid_col : np.ndarray
        Source row (``y``) and column (``x``) coordinates at the grid nodes.

    grid_resolution : tuple[int, int], optional
        Oversampling factors of the grid ``(rows, columns)``.

    win : np.ndarray or None, optional
        Window of the full resolution grid, inclusive bounds. Only the grid
        nodes needed to cover it are used. None uses the whole grid.

    grid_mask : np.ndarray or None, optional
        Integer mask of the grid nodes, of type ``uint8``. Exclusive with
        ``grid_nodata``.

    grid_mask_valid_value : int or None, optional
        Value of ``grid_mask`` designating the valid nodes.

    grid_nodata : float or None, optional
        Row coordinate value designating the invalid nodes. Exclusive with
        ``grid_mask``.

    mode : {"longest-distance", "least-square-fit"}, optional
        Estimation mode.

    Returns
    -------
    np.ndarray
        The 2x2 matrix ``[w1 w2]``.

    Raises
    ------
    ValueError
        If the mode is unknown, the validity arguments are inconsistent, or the
        valid nodes do not define ``W``.
    """
    _check_node_validity_args(grid_mask, grid_mask_valid_value, grid_nodata)
    nodes = {
        "grid_resolution": grid_resolution,
        "win": win,
        "grid_mask": grid_mask,
        "grid_mask_valid_value": grid_mask_valid_value,
        "grid_nodata": grid_nodata,
    }
    if mode == "longest-distance":
        return _longest_distance_matrix(grid_row, grid_col, **nodes)
    if mode == "least-square-fit":
        return fit_affine_least_squares(grid_row, grid_col, **nodes)[0]
    raise ValueError(
        f"Unknown transition matrix estimation mode {mode!r}, expected "
        "'longest-distance' or 'least-square-fit'"
    )


# --------------------------------------------------------------------------- #
# Resolution of a filter specification
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GridAntialiasingDiagnostics:
    """
    Quantities computed while resolving a filter specification on a grid.

    Attributes
    ----------
    transition_matrix : np.ndarray
        Estimated transition matrix ``W``.

    transition_matrix_estimation : str
        Estimation mode used.

    affine_residual : float
        Largest distance, in source pixels, between a valid grid node and the
        least-squares affine model. A large value means that one kernel is a
        coarse description of the grid.

    filter : Any
        Diagnostics of the method, for instance
        :class:`~gridr.core.filter.antialiasing.reciprocal_cell.ReciprocalCellDiagnostics`.
    """

    transition_matrix: np.ndarray
    transition_matrix_estimation: str
    affine_residual: float
    filter: Any


def resolve_antialiasing_filter(
    spec: AntialiasingFilter | KernelFilter | None,
    grid_row: np.ndarray,
    grid_col: np.ndarray,
    grid_resolution: tuple[int, int] = (1, 1),
    win: np.ndarray | None = None,
    grid_mask: np.ndarray | None = None,
    grid_mask_valid_value: int | None = 1,
    grid_nodata: float | None = None,
    transition_matrix_estimation: str = "longest-distance",
) -> tuple[KernelFilter | None, GridAntialiasingDiagnostics | None]:
    """
    Resolve an antialiasing filter specification on the whole grid.

    Parameters
    ----------
    spec : AntialiasingFilter, KernelFilter or None
        Specification. A :class:`~gridr.core.filter.antialiasing.KernelFilter`
        is returned unchanged.

    grid_row, grid_col : np.ndarray
        Backward grid at its nodes: source row (``y``) and column (``x``)
        coordinates.

    grid_resolution : tuple[int, int], optional
        Oversampling factors of the grid ``(rows, columns)``.

    win : np.ndarray or None, optional
        Window of the full resolution grid, ``[[first_row, last_row],
        [first_col, last_col]]`` with inclusive bounds, as in
        :func:`~gridr.core.grid.grid_resampling.array_grid_resampling`. Only
        the grid nodes needed to cover it are used. None uses the whole grid.

    grid_mask : np.ndarray or None, optional
        Integer mask of the grid nodes, of type ``uint8``. Nodes whose value
        differs from ``grid_mask_valid_value`` are left out. Exclusive with
        ``grid_nodata``.

    grid_mask_valid_value : int or None, optional
        Value of ``grid_mask`` designating the valid nodes.

    grid_nodata : float or None, optional
        Row coordinate value designating the invalid nodes. Exclusive with
        ``grid_mask``.

    transition_matrix_estimation : {"longest-distance", "least-square-fit"}, optional
        Estimation of the transition matrix ``W`` (see
        :func:`estimate_transition_matrix`).

    Returns
    -------
    tuple[KernelFilter or None, GridAntialiasingDiagnostics or None]
        The filter, None when no filtering is requested or needed, and the
        diagnostics (None unless a specification was designed).

    Raises
    ------
    TypeError
        If ``spec`` is neither None, a ``KernelFilter`` nor an
        ``AntialiasingFilter``.

    ValueError
        If the estimation mode is unknown, the validity arguments are
        inconsistent, or the valid nodes do not define ``W``.
    """
    # TODO: one kernel for the whole grid; a large affine_residual calls for a
    # design per tile.
    # TODO: not yet called by array_grid_resampling nor by the chains.
    if spec is None:
        return None, None
    if isinstance(spec, KernelFilter):
        return spec, None
    if not isinstance(spec, AntialiasingFilter):
        raise TypeError(
            "spec must be None, a KernelFilter or an AntialiasingFilter, "
            f"got {type(spec).__name__}"
        )
    nodes = {
        "grid_resolution": grid_resolution,
        "win": win,
        "grid_mask": grid_mask,
        "grid_mask_valid_value": grid_mask_valid_value,
        "grid_nodata": grid_nodata,
    }
    if transition_matrix_estimation == "least-square-fit":
        w, _, residual = fit_affine_least_squares(grid_row, grid_col, **nodes)
    else:
        w = estimate_transition_matrix(
            grid_row, grid_col, mode=transition_matrix_estimation, **nodes
        )
        residual = fit_affine_least_squares(grid_row, grid_col, **nodes)[2]
    kf, diag = spec.design(w)
    return kf, GridAntialiasingDiagnostics(
        transition_matrix=w,
        transition_matrix_estimation=transition_matrix_estimation,
        affine_residual=residual,
        filter=diag,
    )


# --------------------------------------------------------------------------- #
# Deprecated interface
# --------------------------------------------------------------------------- #
class ComputeAntialiasingFilterStatus(Enum):
    """Enum class to represent `compute_antialiasing_filter_from_grid` status"""

    NA = 0
    NOT_REQUIRED = 1
    REQUIRED = 2
    REQUIRED_FOR_ZOOM = 3


def compute_antialiasing_filter_from_grid(
    grid_row: np.ndarray,
    grid_col: np.ndarray,
    grid_resolution: tuple[int, int],
    filter_nrow: int,
    filter_ncol: int,
    filter_cutoff_tanh_slope: float | None = 10.0,
    filter_cutoff_shift: float = 0.0,
    win: np.ndarray | None = None,
    grid_mask: np.ndarray | None = None,
    grid_mask_valid_value: int | None = 1,
    grid_nodata: float | None = None,
) -> tuple[ComputeAntialiasingFilterStatus, np.ndarray | None]:
    """
    Create an antialiasing filter adapted to the grid geometric transformation.

    .. deprecated:: 0.8.0
        This function is a wrapper around
        :func:`resolve_antialiasing_filter` with a
        :class:`~gridr.core.filter.antialiasing.reciprocal_cell.ReciprocalCellFilter`,
        kept for one version. The parameters are translated as follows:

        - ``filter_nrow`` and ``filter_ncol`` must be equal and odd; they give
          the forced ``kernel_size``;
        - ``filter_cutoff_tanh_slope`` gives ``cutoff_tanh_slope``; None (hard
          gate) is no longer supported;
        - ``filter_cutoff_shift = s`` gives the ``aliasing_tolerance`` equal to
          the gate value on the cell boundary for a half-width ``1/2 + s``;
          ``s`` must not be positive;
        - ``win``, ``grid_mask``, ``grid_mask_valid_value`` and
          ``grid_nodata`` are passed unchanged. ``win`` is a window of the
          full resolution grid, as documented; the previous implementation
          passed it to the grid geometry analysis as a window of grid nodes,
          which differs when ``grid_resolution`` is not ``(1, 1)``.

        The transition matrix is estimated with ``"longest-distance"`` and the
        cell is built in the destination metric, as before.

    Parameters
    ----------
    grid_row, grid_col : np.ndarray
        Row and column coordinates of the target grid in the input array.

    grid_resolution : tuple[int, int]
        Oversampling factors of the grid for rows and columns.

    filter_nrow, filter_ncol : int
        Size of the output filter; both must be equal and odd.

    filter_cutoff_tanh_slope : float or None, default 10.
        Slope of the tanh transition.

    filter_cutoff_shift : float, default 0.
        Shift of the transition, zero or negative.

    win : np.ndarray or None, default None
        Window of the full resolution grid, inclusive bounds.

    grid_mask : np.ndarray or None, default None
        Integer mask of the grid nodes, of type ``uint8``.

    grid_mask_valid_value : int or None, default 1
        Value of ``grid_mask`` designating valid nodes.

    grid_nodata : float or None, default None
        Row coordinate value designating invalid nodes.

    Returns
    -------
    tuple[ComputeAntialiasingFilterStatus, np.ndarray or None]
        ``NOT_REQUIRED`` and None when no filter is needed, otherwise
        ``REQUIRED`` and the kernel.

    Raises
    ------
    ValueError
        If the parameters cannot be translated (hard gate, positive shift,
        different or even sizes).
    """
    # TODO: remove in 0.9.0, with ComputeAntialiasingFilterStatus.
    warnings.warn(
        "compute_antialiasing_filter_from_grid is deprecated, use "
        "resolve_antialiasing_filter with a ReciprocalCellFilter instead",
        DeprecationWarning,
        stacklevel=2,
    )
    if filter_cutoff_tanh_slope is None:
        raise ValueError(
            "A hard gate (filter_cutoff_tanh_slope=None) is no longer supported"
        )
    if filter_cutoff_shift > 0:
        raise ValueError("A positive filter_cutoff_shift is no longer supported")
    if filter_nrow != filter_ncol:
        raise ValueError("filter_nrow and filter_ncol must be equal")
    tolerance = float(
        tanh_gate(np.array(0.5), filter_cutoff_tanh_slope, 0.5 + filter_cutoff_shift)
    )
    spec = ReciprocalCellFilter(
        cutoff_tanh_slope=filter_cutoff_tanh_slope,
        aliasing_tolerance=min(tolerance, 0.5),
        kernel_size=filter_nrow,
        isotropic_frame="destination",
    )
    kf, _ = resolve_antialiasing_filter(
        spec,
        grid_row,
        grid_col,
        grid_resolution,
        win=win,
        grid_mask=grid_mask,
        grid_mask_valid_value=grid_mask_valid_value,
        grid_nodata=grid_nodata,
        transition_matrix_estimation="longest-distance",
    )
    if kf is None:
        return ComputeAntialiasingFilterStatus.NOT_REQUIRED, None
    return ComputeAntialiasingFilterStatus.REQUIRED, kf.kernel
