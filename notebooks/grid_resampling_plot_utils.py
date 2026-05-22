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
Plotting and input-generation utilities for the GridR resampling validation
notebook(s).

This module bundles three groups of helpers:

1. Plotting primitives and styling (``PLOT_STYLE``, colormap legend handlers,
   raster / grid / window rendering helpers).
2. The top-level driver :func:`main_plot`, which lays out three diagnostic
   figures side by side: (a) the resampling inputs, (b) the standalone
   preprocessing intermediate state, and (c) the resampling outputs.
3. Test-input generators :func:`_mono_point_grid_generate_input` and
   :func:`_multi_points_grid_generate_input`, which build the keyword
   argument dictionaries consumed by :func:`array_grid_resampling`.
"""

from __future__ import annotations

import copy
from functools import partial
from typing import Any, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.legend_handler import HandlerBase

from gridr.core.grid.grid_resampling import (
    array_grid_resampling,
    calculate_source_extent,
    standalone_preprocessing,
    GridMetricsError,
)
from gridr.core.grid.grid_commons import grid_full_resolution_shape
from gridr.core.interp.interpolator import get_interpolator
from gridr.core.grid.grid_utils import build_grid
from gridr.core.grid.grid_mask import build_mask
from gridr.core.utils.array_window import window_apply


# =============================================================================
# Style definitions
# =============================================================================

# Binary mask colormap: red = invalid, green = valid.
cmap_mask = LinearSegmentedColormap.from_list(
    'custom_rg',
    [(0.0, 'red'), (1.0, 'green')],
)


class ColormapPatch(mpatches.Patch):
    """Legend handle that displays a horizontal colormap strip.

    Used in matplotlib legends to advertise the meaning of a continuous
    colormap (here, the binary mask validity colormap).
    """

    def __init__(self, cmap, n=256, **kwargs):
        self.cmap = cmap
        self.n = n
        super().__init__(**kwargs)


class HandlerColormap(HandlerBase):
    """Legend handler that paints ``ColormapPatch`` as N adjacent rectangles."""

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        patches = []
        w = width / orig_handle.n
        for i in range(orig_handle.n):
            color = orig_handle.cmap(i / (orig_handle.n - 1))
            patch = mpatches.Rectangle(
                [xdescent + i * w, ydescent],
                w, height,
                facecolor=color,
                edgecolor='none',
                transform=trans,
            )
            patches.append(patch)
        return patches


DEFAULT_MARKER_SIZE = 16

# Global style sheet. Centralising styles makes it easy to keep all three
# diagnostic figures visually consistent.
PLOT_STYLE = {
    'coord_grid': {'color': 'black', 'alpha': 0.5, 'linewidth': 0.5, 'linestyle': '-'},
    'imshow_array_in': {'cmap': 'grey', 'alpha': 0.7},
    'imshow_array_in_mask': {'cmap': 'RdYlGn', 'alpha': 0.5, 'vmin': 0, 'vmax': 1},

    # Grid node markers -- "full_res" variants correspond to oversampled
    # (interpolated) nodes when grid_resolution != (1, 1).
    'grid_scatter': {'marker': 'o', 's': DEFAULT_MARKER_SIZE, 'color': 'orange'},
    'grid_scatter_win_out': {'marker': 'o', 's': DEFAULT_MARKER_SIZE, 'color': 'orangered', 'facecolors': 'none'},
    'grid_scatter_invalid': {'marker': 'D', 's': DEFAULT_MARKER_SIZE, 'color': 'red'},
    'grid_scatter_invalid_win_out': {'marker': 'D', 's': DEFAULT_MARKER_SIZE, 'color': 'red', 'facecolors': 'none'},

    'grid_scatter_full_res': {'marker': 'o', 's': DEFAULT_MARKER_SIZE, 'color': 'sandybrown', 'alpha': 0.6},
    'grid_scatter_win_out_full_res': {'marker': 'o', 's': DEFAULT_MARKER_SIZE, 'color': 'orangered', 'facecolors': 'none', 'alpha': 0.6},
    'grid_scatter_invalid_full_res': {'marker': 'D', 's': DEFAULT_MARKER_SIZE, 'color': 'sandybrown', 'alpha': 0.6},
    'grid_scatter_invalid_win_out_full_res': {'marker': 'D', 's': DEFAULT_MARKER_SIZE, 'color': 'orangered', 'facecolors': 'none', 'alpha': 0.6},

    # Grid wireframe (connectors between adjacent nodes).
    'grid_wireframe': {'linewidth': 1, 'linestyle': '-', 'alpha': 0.7, 'color': 'orange'},
    'grid_wireframe_full_res': {'linewidth': 0.5, 'linestyle': '--', 'alpha': 0.7, 'color': 'sandybrown'},

    # Rectangles framing source footprint, read window and padded support.
    'array_in_rect': {'edgecolor': 'royalblue', 'facecolor': 'none', 'linewidth': 2, 'linestyle': '-', 'alpha': 1.0},
    'array_in_win_read': {'edgecolor': 'royalblue', 'facecolor': 'royalblue', 'linewidth': 2, 'linestyle': '--', 'alpha': 0.5},
    'array_in_win_marged': {'edgecolor': 'steelblue', 'facecolor': 'steelblue', 'linewidth': 2, 'linestyle': '--', 'alpha': 0.6},

    # Safe (fully valid) region of the input mask.
    'mask_safe_region': {'edgecolor': 'green', 'facecolor': 'green', 'fill': False, 'linewidth': 1, 'linestyle': '-', 'alpha': 0.8, 'hatch': '//'},
}


# =============================================================================
# Geometric helpers (grid hull, window <-> rectangle conversions)
# =============================================================================

def get_grid_hull(grid_row, grid_col, grid_mask):
    """Return the integer-pixel bounding box that encloses a grid.

    Returns
    -------
    np.ndarray of shape (2, 2)
        ``[[row_min, row_max], [col_min, col_max]]`` with min floored and
        max ceiled to integers (i.e. the pixel hull, not the float hull).
    """
    grid_min_row = int(np.floor(np.min(grid_row)))
    grid_max_row = int(np.ceil(np.max(grid_row)))
    grid_min_col = int(np.floor(np.min(grid_col)))
    grid_max_col = int(np.ceil(np.max(grid_col)))
    return np.asarray((
        (grid_min_row, grid_max_row),
        (grid_min_col, grid_max_col),
    ))


def get_grid_hull_margin(grid_row, grid_col, grid_mask, src_shape):
    """Compute how far the grid hull extends beyond the source raster.

    The result is a non-negative padding ``((top, bottom), (left, right))``
    expressed in source pixels. A zero entry means the grid does not exceed
    the raster on that side.
    """
    h = get_grid_hull(grid_row, grid_col, grid_mask)
    pad = (
        (max(0, -h[0, 0]), max(0, h[0, 1] - src_shape[0])),
        (max(0, -h[1, 0]), max(0, h[1, 1] - src_shape[1])),
    )
    return np.asarray(pad)


def from_extent(win, extent):
    """Map a (row, col) window into matplotlib axes coordinates.

    Parameters
    ----------
    win : array-like of shape (2, 2)
        ``[[row_start, row_end], [col_start, col_end]]`` (inclusive).
    extent : tuple
        Matplotlib-style ``(left, right, bottom, top)`` extent of the host
        raster.

    Returns
    -------
    tuple of float
        ``(rect_left, rect_right, rect_bottom, rect_top)``.
    """
    ext_left, _, _, ext_top = extent

    rect_left = ext_left + win[1, 0]
    rect_right = ext_left + win[1, 1] + 1
    rect_top = ext_top + win[0, 0]
    rect_bottom = ext_top + win[0, 1] + 1
    return (rect_left, rect_right, rect_bottom, rect_top)


def from_extent_array_origin(array_origin, shape, extent):
    """Map an array described by its origin and shape to axes coordinates.

    Convenience wrapper around :func:`from_extent` that builds the window
    from an ``array_origin`` (offset of the array's (0, 0) within the source
    coordinate frame) and a ``shape``.
    """
    if len(shape) == 3:
        shape = (shape[1], shape[2])
    win = (
        (-array_origin[0], -array_origin[0] + shape[0] - 1),
        (-array_origin[1], -array_origin[1] + shape[1] - 1),
    )
    return from_extent(np.asarray(win), extent)


def win_to_rect(win, extent):
    """Return ``(xy, width, height)`` ready for ``mpatches.Rectangle``."""
    rect_left, rect_right, rect_bottom, rect_top = from_extent(win, extent)
    return ((rect_left, rect_top), rect_right - rect_left, rect_bottom - rect_top)


# =============================================================================
# Low-level plotting helpers
# =============================================================================

def _plot_raster_pixel(ax, raster, extent, pixel_grid_shape, margin,
                       binary=False, **style):
    """Render a raster with explicit pixel boundaries and centered ticks.

    The function imshows the array, then draws a thin grid on every pixel
    boundary and offsets tick labels so the (0, 0) label sits at the source
    raster origin even when a margin is displayed.
    """
    imshow_style = 'imshow_array_in_mask' if binary else 'imshow_array_in'
    ax.imshow(raster, origin='upper', extent=extent,
              aspect='equal', **style[imshow_style])

    # Axis limits aligned to pixel boundaries.
    ax.set_xlim(-0.5, pixel_grid_shape[1] - 0.5)
    ax.set_ylim(pixel_grid_shape[0] - 0.5, -0.5)

    # Pixel boundary lines.
    for col in range(pixel_grid_shape[1] + 1):
        ax.axvline(x=col - 0.5, **style['coord_grid'])
    for row in range(pixel_grid_shape[0] + 1):
        ax.axhline(y=row - 0.5, **style['coord_grid'])

    # Ticks at pixel centers, labels shifted so origin matches source.
    x_ticks = np.arange(pixel_grid_shape[1])
    y_ticks = np.arange(pixel_grid_shape[0])
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels(x_ticks - margin[1, 0], fontsize=FS_TICK)
    ax.set_yticklabels(y_ticks - margin[0, 0], fontsize=FS_TICK)


def _legend_section_title(label):
    """Build an invisible patch used as a section header in legends."""
    return mpatches.Patch(visible=False, label=label)


def _legend_separator():
    """Build an invisible Line2D used as a blank separator in legends."""
    return Line2D([0], [0], visible=False, label='')


def _legend_scatter_to_line2d_kwargs(kwargs):
    """Translate scatter-style kwargs to Line2D-compatible keys.

    Unused at the moment but kept around for legend handles that should
    appear as a single marker (matplotlib Line2D expects ``markersize`` and
    ``markerfacecolor`` rather than the scatter ``s`` / ``facecolors``).
    """
    k = copy.copy(kwargs)
    try:
        k['markersize'] = k.pop('s')
    except KeyError:
        pass
    try:
        k['markerfacecolor'] = k.pop('facecolors')
    except KeyError:
        pass
    return k


def _plot_array_box(ax, extent, shape, label, **style):
    """Draw a rectangle of the given shape at the given extent."""
    if len(shape) == 3:
        shape = (shape[1], shape[2])
    ax.add_patch(mpatches.Rectangle(
        (extent[0], extent[2]),
        shape[1], -shape[0],
        label=label, **style,
    ))


# =============================================================================
# Figure layout
# =============================================================================

DPI = 110

# Default rendering DPI. 110 dpi is large enough to be sharp on screen and
# small enough that font-pixel-size is reasonable.

# Target inches per source pixel. The actual value is adaptively shrunk for
# large rasters (see ``_compute_panel_size``) so a 5x5 grid stays readable
# while a 50x60 grid still fits on screen.
INCHES_PER_PIXEL_TARGET = 0.25

# Per-panel size clamps (inches). Each raster panel is forced into this
# range; the legend column sits outside this and is sized separately.
PANEL_MIN_IN = 3.0
PANEL_MAX_IN = 7.0

# Inches reserved on the right of the figure for the externally-anchored
# legend (added on top of the raster + mask panels).
LEGEND_PANEL_IN = 3.2

# Fixed font sizes for every figure. Picking absolute pt values makes the
# rendering reproducible regardless of figsize, which was the main source
# of "tiny labels on small figures, huge labels on big ones" before.
FS_SUPTITLE = 12
FS_TITLE = 10
FS_LEGEND = 8
FS_LEGEND_TITLE = 9
FS_TICK = 8

# Backwards-compatible module constants. ``PIXEL_SIZE`` is no longer used
# internally but is kept so older notebooks that still pass it as a kwarg
# do not crash.
PIXEL_SIZE = 80


def _compute_panel_size(nrows_tot, ncols_tot):
    """Compute the figsize of a single raster panel (inches).

    The size is proportional to the raster dimensions, but with an
    adaptive ``inches_per_pixel`` that shrinks for large rasters so panel
    dimensions stay in ``[PANEL_MIN_IN, PANEL_MAX_IN]``. Aspect ratio is
    preserved as long as both axes fit within the clamps; otherwise the
    binding axis is anchored at the clamp and the other follows the
    aspect.
    """
    max_dim = max(nrows_tot, ncols_tot)

    # Adaptive inches-per-pixel: linear shrink between 20 and 60 pixels.
    if max_dim <= 20:
        ipp = INCHES_PER_PIXEL_TARGET
    elif max_dim >= 60:
        ipp = INCHES_PER_PIXEL_TARGET * 0.4
    else:
        # Linear blend between the two reference points.
        t = (max_dim - 20) / (60 - 20)
        ipp = INCHES_PER_PIXEL_TARGET * (1.0 - 0.6 * t)

    width = ncols_tot * ipp
    height = nrows_tot * ipp

    # Clamp the binding dimension, scale the other to preserve aspect.
    aspect = ncols_tot / max(nrows_tot, 1)
    if width > PANEL_MAX_IN:
        width = PANEL_MAX_IN
        height = width / aspect
    if height > PANEL_MAX_IN:
        height = PANEL_MAX_IN
        width = height * aspect
    if width < PANEL_MIN_IN:
        width = PANEL_MIN_IN
        height = max(width / aspect, PANEL_MIN_IN)
    if height < PANEL_MIN_IN:
        height = PANEL_MIN_IN
        width = max(height * aspect, PANEL_MIN_IN)

    return (width, height)


def _plot_resampling_figsize_per_ax(
    interp, array_in,
    grid_row, grid_col, grid_resolution,
    nodata_out, array_in_origin, win,
    array_in_mask, array_in_mask_safe_win,
    grid_mask, grid_mask_valid_value, grid_nodata,
    check_boundaries, standalone, boundary_condition, trust_padding,
    plot_margin=1, pixel_size=None, dpi=None,
):
    """Compute figure size, raster extent and display margin for one panel.

    The display margin is enlarged to cover both the grid hull excess and
    the interpolator support padding so the panel shows all data the
    resampling will read, including out-of-source extrapolation regions.

    ``pixel_size`` and ``dpi`` are accepted for backwards compatibility
    but no longer drive the figure size; sizing is now handled by
    :func:`_compute_panel_size`.
    """
    margin = np.asarray(
        [[plot_margin, plot_margin], [plot_margin, plot_margin]]
    )

    has_grid = grid_row is not None and grid_col is not None
    pad = None

    nrows, ncols = array_in.shape

    if has_grid:
        # Padding required by the interpolator support window.
        (_, _, pad, _) = calculate_source_extent(
            interp=interp,
            array_in=array_in,
            grid_row=grid_row,
            grid_col=grid_col,
            grid_resolution=grid_resolution,
            grid_nodata=grid_nodata,
            grid_mask=grid_mask,
            grid_mask_valid_value=grid_mask_valid_value,
            win=win,
            safecheck_src_boundaries=True,
            logger_msg_prefix=None,
            logger=None,
        )
        # Padding required by the grid hull itself.
        grid_hull_pad = get_grid_hull_margin(
            grid_row, grid_col, grid_mask, array_in.shape,
        )

        # Display margin = max(interpolator pad, grid hull pad).
        max_pad = grid_hull_pad
        if pad is not None:
            max_pad = np.maximum(pad, grid_hull_pad)
        margin += max_pad

    nrows_tot = nrows + np.sum(margin[0])
    ncols_tot = ncols + np.sum(margin[1])
    raster_extent = [
        -0.5 + margin[1][0], ncols_tot - margin[1][1] - 0.5,
        nrows_tot - margin[0][1] - 0.5, margin[0][0] - 0.5,
    ]

    figsize = _compute_panel_size(nrows_tot, ncols_tot)
    return figsize, raster_extent, margin


# =============================================================================
# Grid wireframe overlay
# =============================================================================

def _plot_grid_wire(
    _ax, handles,
    grid_row, grid_col, grid_mask, grid_mask_valid_value,
    resolution, win, margin,
    **style,
):
    """Overlay the resampling grid (node markers + node connectors).

    When the grid resolution is greater than (1, 1) the fully-resolved grid
    is computed via :func:`build_grid` and the in-between (interpolated)
    nodes are drawn with the *_full_res* style variants.

    When a grid mask is provided, nodes are split into valid / invalid;
    when a computation window ``win`` is provided, nodes are further split
    into in-window / out-of-window. Every category gets its own legend
    entry.
    """
    if handles is None:
        handles = []
    if grid_row is None or grid_col is None:
        return

    handles.extend([
        _legend_separator(),
        _legend_section_title(' - Grid - '),
    ])

    # Full-resolution version of the grid (resolution == (1, 1)).
    grid_row_full_res, grid_col_full_res = build_grid(
        resolution=(1, 1),
        grid=np.asarray([grid_row, grid_col]),
        grid_target_win=None,  # forced to None to compute the whole grid
        grid_resolution=resolution,
        out=None,
        computation_dtype=np.float64,
    )

    # Mark which full-res nodes correspond to original (non-oversampled) nodes.
    grid_orig_mask = np.zeros(grid_row_full_res.shape, dtype=np.uint8)
    grid_orig_mask[::resolution[0], ::resolution[1]] = 1

    # Resample the (binary) grid mask to full resolution as well, so we know
    # which interpolated nodes inherit valid / invalid status.
    grid_valid_mask = np.ones(grid_row_full_res.shape, dtype=np.float32)
    if grid_mask is not None and np.any(grid_mask != grid_mask_valid_value):
        build_mask(
            shape=grid_valid_mask.shape,
            resolution=(1, 1),
            out=grid_valid_mask,
            geometry_origin=None,
            geometry_pair=None,
            mask_in=grid_mask,
            mask_in_target_win=None,
            mask_in_resolution=resolution,
            oversampling_dtype=grid_valid_mask.dtype,
            mask_in_binary_threshold=0.99999999,
            rasterize_kwargs=None,
            init_out=False,
        )
    grid_valid_mask = grid_valid_mask.astype(np.uint8)

    # -------------------------------------------------------------
    # Wireframe (lines connecting adjacent nodes)
    # -------------------------------------------------------------
    if resolution[0] != 1 or resolution[1] != 1:
        # Full-resolution connectors (between interpolated nodes).
        if resolution[0] != 1:
            for irow in range(grid_row_full_res.shape[0]):
                if irow % resolution[1] != 0:
                    _ax.plot(
                        grid_col_full_res[irow, :] + margin[1, 0],
                        grid_row_full_res[irow, :] + margin[0, 0],
                        **style['grid_wireframe_full_res'],
                    )
        if resolution[1] != 1:
            for icol in range(grid_row_full_res.shape[1]):
                if icol % resolution[1] != 0:
                    _ax.plot(
                        grid_col_full_res[:, icol] + margin[1, 0],
                        grid_row_full_res[:, icol] + margin[0, 0],
                        **style['grid_wireframe_full_res'],
                    )
        handles.append(
            Line2D([0], [0], **style['grid_wireframe_full_res'],
                   label="full resolute grid node's connectors")
        )

    # Connectors between original (provided) grid nodes.
    for irow in range(grid_row.shape[0]):
        _ax.plot(
            grid_col[irow, :] + margin[1, 0],
            grid_row[irow, :] + margin[0, 0],
            **style['grid_wireframe'],
        )
    for icol in range(grid_row.shape[1]):
        _ax.plot(
            grid_col[:, icol] + margin[1, 0],
            grid_row[:, icol] + margin[0, 0],
            **style['grid_wireframe'],
        )
    handles.append(
        Line2D([0], [0], **style['grid_wireframe'],
               label="grid node's connectors")
    )

    # -------------------------------------------------------------
    # Scatter plots of node markers
    # -------------------------------------------------------------
    # For oversampled (full-res) nodes only.
    if resolution[0] != 1 or resolution[1] != 1:
        if win is None:
            indices = np.where(grid_orig_mask == 0)
            handles.append(
                _ax.scatter(
                    grid_col_full_res[indices].flatten() + margin[1, 0],
                    grid_row_full_res[indices].flatten() + margin[0, 0],
                    **style['grid_scatter_full_res'],
                    label="grid node's",
                )
            )
        else:
            # Window mask for full-res grid (which nodes lie in the
            # computation window).
            win_mask = np.zeros(grid_row_full_res.shape, dtype=np.uint8)
            win_mask_view = window_apply(win_mask, win)
            win_mask_view[...] = 1

            # Four categories: (in/out of window) x (valid/invalid).
            iv_v = np.where((win_mask == 1) & (grid_orig_mask == 0) & (grid_valid_mask == 1))
            ov_v = np.where((win_mask == 0) & (grid_orig_mask == 0) & (grid_valid_mask == 1))
            iv_i = np.where((win_mask == 1) & (grid_orig_mask == 0) & (grid_valid_mask == 0))
            ov_i = np.where((win_mask == 0) & (grid_orig_mask == 0) & (grid_valid_mask == 0))

            handles.extend([
                _ax.scatter(
                    grid_col_full_res[iv_v].flatten() + margin[1, 0],
                    grid_row_full_res[iv_v].flatten() + margin[0, 0],
                    **style['grid_scatter_full_res'],
                    label="oversampled grid valid node's",
                ),
                _ax.scatter(
                    grid_col_full_res[ov_v].flatten() + margin[1, 0],
                    grid_row_full_res[ov_v].flatten() + margin[0, 0],
                    **style['grid_scatter_win_out_full_res'],
                    label="oversampled grid valid node's - out of computing window",
                ),
                _ax.scatter(
                    grid_col_full_res[iv_i].flatten() + margin[1, 0],
                    grid_row_full_res[iv_i].flatten() + margin[0, 0],
                    **style['grid_scatter_invalid_full_res'],
                    label="oversampled grid invalid node's",
                ),
                _ax.scatter(
                    grid_col_full_res[ov_i].flatten() + margin[1, 0],
                    grid_row_full_res[ov_i].flatten() + margin[0, 0],
                    **style['grid_scatter_invalid_win_out_full_res'],
                    label="oversampled grid invalid node's - out of computing window",
                ),
            ])

    # Original (provided) grid nodes.
    if win is None:
        indices_valid = np.where((grid_orig_mask == 1) & (grid_valid_mask == 1))
        indices_invalid = np.where((grid_orig_mask == 1) & (grid_valid_mask == 0))
        handles.extend([
            _ax.scatter(
                grid_col_full_res[indices_valid].flatten() + margin[1, 0],
                grid_row_full_res[indices_valid].flatten() + margin[0, 0],
                **style['grid_scatter'],
                label="grid valid node's",
            ),
            _ax.scatter(
                grid_col_full_res[indices_invalid].flatten() + margin[1, 0],
                grid_row_full_res[indices_invalid].flatten() + margin[0, 0],
                **style['grid_scatter_invalid'],
                label="grid invalid node's",
            ),
        ])
    else:
        win_mask = np.zeros(grid_row_full_res.shape, dtype=np.uint8)
        win_mask_view = window_apply(win_mask, win)
        win_mask_view[...] = 1

        iv_v = np.where((win_mask == 1) & (grid_orig_mask == 1) & (grid_valid_mask == 1))
        ov_v = np.where((win_mask == 0) & (grid_orig_mask == 1) & (grid_valid_mask == 1))
        iv_i = np.where((win_mask == 1) & (grid_orig_mask == 1) & (grid_valid_mask == 0))
        ov_i = np.where((win_mask == 0) & (grid_orig_mask == 1) & (grid_valid_mask == 0))

        handles.extend([
            _ax.scatter(
                grid_col_full_res[iv_v].flatten() + margin[1, 0],
                grid_row_full_res[iv_v].flatten() + margin[0, 0],
                **style['grid_scatter'],
                label="grid valid node's",
            ),
            _ax.scatter(
                grid_col_full_res[ov_v].flatten() + margin[1, 0],
                grid_row_full_res[ov_v].flatten() + margin[0, 0],
                **style['grid_scatter_win_out'],
                label="grid valid node's - out of computing window",
            ),
            _ax.scatter(
                grid_col_full_res[iv_i].flatten() + margin[1, 0],
                grid_row_full_res[iv_i].flatten() + margin[0, 0],
                **style['grid_scatter_invalid'],
                label="grid invalid node's",
            ),
            _ax.scatter(
                grid_col_full_res[ov_i].flatten() + margin[1, 0],
                grid_row_full_res[ov_i].flatten() + margin[0, 0],
                **style['grid_scatter_invalid_win_out'],
                label="grid invalid node's - out of computing window",
            ),
        ])


# =============================================================================
# Panel plotting: inputs, preprocessing, outputs
# =============================================================================

def _attach_external_legend(fig, handles, title=None):
    """Anchor a single legend in the right margin of the figure, outside subplots.

    The figure is sized by :func:`_build_figure` to reserve a strip of
    width ``LEGEND_PANEL_IN`` on the right; the legend is anchored at
    the inner edge of that strip so it stays fully inside the figure
    boundary (visible even when the renderer does not crop the bbox).
    """
    fig_w_in = fig.get_size_inches()[0]
    legend_fraction = LEGEND_PANEL_IN / fig_w_in
    leg = fig.legend(
        handles=handles,
        handler_map={ColormapPatch: HandlerColormap()},
        loc='center left',
        # Anchor just inside the reserved right strip.
        bbox_to_anchor=(1.0 - legend_fraction + 0.005, 0.5),
        bbox_transform=fig.transFigure,
        fontsize=FS_LEGEND,
        frameon=True, framealpha=0.95, edgecolor='gray',
        borderaxespad=0.5,
        title=title,
        title_fontsize=FS_LEGEND_TITLE,
    )
    return leg


def _build_figure(panel_size, suptitle):
    """Create a 1x2 figure (raster + mask) with room for an external legend.

    Returns ``(fig, axes)`` where ``axes`` is a length-2 array of axes
    already in their final position. The figure uses
    ``constrained_layout`` so titles, ticks and the right-side legend
    are spaced consistently.
    """
    panel_w, panel_h = panel_size
    # Width = 2 raster panels + space reserved for the right-anchored legend.
    fig_w = 2 * panel_w + LEGEND_PANEL_IN
    fig_h = panel_h + 0.6  # small extra room for suptitle / ticks
    fig, axes = plt.subplots(
        1, 2, figsize=(fig_w, fig_h), dpi=DPI,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 1]},
    )
    # Reserve room on the right for the external legend (fraction of width).
    legend_fraction = LEGEND_PANEL_IN / fig_w
    fig.get_layout_engine().set(
        w_pad=0.04, h_pad=0.04, wspace=0.04, hspace=0.04,
        rect=(0.0, 0.0, 1.0 - legend_fraction, 1.0),
    )
    fig.suptitle(suptitle, fontsize=FS_SUPTITLE, fontweight='bold')
    return fig, axes


def _plot_resampling_inputs(
    fig, axes, extent, margin,
    interp, array_in,
    grid_row, grid_col, grid_resolution,
    nodata_out, array_in_origin, win,
    array_in_mask, array_in_mask_safe_win,
    grid_mask, grid_mask_valid_value, grid_nodata,
    check_boundaries, standalone, boundary_condition, trust_padding,
    style=PLOT_STYLE,
):
    """Render the *inputs* panel: source raster + input mask.

    The legend is built incrementally into ``handles`` and attached to
    ``fig`` (outside the panels) at the end of the function.
    """
    nrows, ncols = array_in.shape
    nrows_tot = nrows + np.sum(margin[0])
    ncols_tot = ncols + np.sum(margin[1])

    # NB: calculate_source_extent is invoked for parity with the
    # preprocessing panel; only its grid-window outputs are used to decorate
    # the input figure if needed. Currently the read / marged windows are
    # plotted only in the preprocessing panel below.
    if grid_row is not None and grid_col is not None:
        calculate_source_extent(
            interp=interp,
            array_in=array_in,
            grid_row=grid_row,
            grid_col=grid_col,
            grid_resolution=grid_resolution,
            grid_nodata=None,
            grid_mask=grid_mask,
            grid_mask_valid_value=grid_mask_valid_value,
            win=win,
            safecheck_src_boundaries=True,
            logger_msg_prefix=None,
            logger=None,
        )

    # ------------------------------------------------------------------
    # Source raster
    # ------------------------------------------------------------------
    ax = axes[0]
    handles = [_legend_section_title(' - Input Image - ')]
    _plot_raster_pixel(ax, array_in, extent, (nrows_tot, ncols_tot), margin, **style)
    _plot_array_box(ax, extent, array_in.shape, 'Original Zone', **style['array_in_rect'])
    ax.set_title('Input Image', fontsize=FS_TITLE)
    handles.append(mpatches.Patch(**style['array_in_rect'], label='Source Footprint'))

    _plot_grid_wire(ax, handles, grid_row, grid_col, grid_mask,
                    grid_mask_valid_value, grid_resolution, win, margin, **style)

    # ------------------------------------------------------------------
    # Optional input mask
    # ------------------------------------------------------------------
    ax = axes[1]
    if array_in_mask is not None:
        _plot_raster_pixel(ax, array_in_mask, extent,
                           (nrows_tot, ncols_tot), margin, binary=True, **style)
        ax.set_title('Input Mask', fontsize=FS_TITLE)
        _plot_array_box(ax, extent, array_in.shape, 'Original Zone', **style['array_in_rect'])
        handles.extend([
            _legend_separator(),
            _legend_section_title(' - Input Mask - '),
            mpatches.Patch(**style['array_in_rect'], label='Source Footprint'),
            ColormapPatch(cmap=cmap_mask, n=2, label='Mask Validity (novalid/valid)'),
        ])

        if array_in_mask_safe_win is not None:
            safe_region_extent = from_extent(array_in_mask_safe_win, extent)
            safe_region_shape = (
                array_in_mask_safe_win[:, 1] - array_in_mask_safe_win[:, 0] + 1
            )
            _plot_array_box(ax, safe_region_extent, safe_region_shape,
                            'Safe Region', **style['mask_safe_region'])
            handles.append(
                mpatches.Patch(**style['mask_safe_region'],
                               label='Mask Safe Valid Window')
            )

        _plot_grid_wire(ax, None, grid_row, grid_col, grid_mask,
                        grid_mask_valid_value, grid_resolution, win, margin, **style)
    else:
        ax.text(0.5, 0.5, 'no mask', ha='center', va='center',
                fontsize=FS_TITLE, color='gray', style='italic')
        ax.set_title('Input Mask', fontsize=FS_TITLE)
        ax.axis('off')

    # ------------------------------------------------------------------
    # External legend (right of the figure)
    # ------------------------------------------------------------------
    _attach_external_legend(fig, handles)


def _plot_resampling_preprocessing(
    fig, axes, extent, margin,
    interp, array_in,
    grid_row, grid_col, grid_resolution,
    nodata_out, array_in_origin, win,
    array_in_mask, array_in_mask_safe_win,
    grid_mask, grid_mask_valid_value, grid_nodata,
    check_boundaries, standalone, boundary_condition, trust_padding,
    style=PLOT_STYLE,
):
    """Render the *standalone preprocessing* panel.

    Shows the padded / boundary-conditioned array and mask as the
    resampler would actually consume them, together with:
      - the source read window (``array_src_win_read``) -- the minimal
        window into the original source raster required by the grid,
      - the padded support window (``array_src_win_marged``) -- the
        read window extended to host the interpolator support, and
      - the propagated safe region of the mask, if any.
    """
    nrows, ncols = array_in.shape
    nrows_tot = nrows + np.sum(margin[0])
    ncols_tot = ncols + np.sum(margin[1])

    array_src_win_read = array_src_win_marged = pad = grid_metrics = None

    if grid_row is not None and grid_col is not None:
        (
            array_src_win_read,
            array_src_win_marged,
            pad,
            grid_metrics,
        ) = calculate_source_extent(
            interp=interp,
            array_in=array_in,
            grid_row=grid_row,
            grid_col=grid_col,
            grid_resolution=grid_resolution,
            grid_nodata=None,
            grid_mask=grid_mask,
            grid_mask_valid_value=grid_mask_valid_value,
            win=win,
            safecheck_src_boundaries=True,
            logger_msg_prefix=None,
            logger=None,
        )
    print(f"preprocessing:grid_metrics : {grid_metrics}")

    # ------------------------------------------------------------------
    # Run standalone preprocessing
    # ------------------------------------------------------------------
    array_in_shape = array_in.shape
    if len(array_in_shape) == 2:
        array_in_shape = (1,) + array_in_shape

    bypass = False
    try:
        (
            preprocessed_array,
            preprocessed_array_shape,
            preprocessed_array_origin,
            preprocessed_array_mask,
            preprocessed_array_mask_safe_region,
            _,
        ) = standalone_preprocessing(
            interp=interp,
            array_in=array_in,
            array_in_shape=array_in_shape,
            array_in_origin=(0, 0),
            grid_row=grid_row,
            grid_col=grid_col,
            grid_resolution=grid_resolution,
            grid_nodata=None,
            grid_mask=grid_mask,
            grid_mask_valid_value=1,
            win=win,
            array_in_mask=array_in_mask,
            array_in_mask_safe_win=array_in_mask_safe_win,
            boundary_condition=boundary_condition,
            trust_padding=trust_padding,
            check_boundaries=True,
            logger_msg_prefix=None,
            logger=None,
        )
    except GridMetricsError:
        # The grid layout is inconsistent with the source/window: skip
        # rendering rather than crashing the whole figure.
        bypass = True

    if bypass:
        return

    # ------------------------------------------------------------------
    # Preprocessed array
    # ------------------------------------------------------------------
    ax = axes[0]
    handles = [_legend_section_title(' - Standalone Preprocessed Image - ')]

    preprocessed_raster_extent = from_extent_array_origin(
        preprocessed_array_origin, preprocessed_array_shape, extent,
    )
    _plot_raster_pixel(ax, preprocessed_array, preprocessed_raster_extent,
                       (nrows_tot, ncols_tot), margin, **style)
    _plot_array_box(ax, extent, array_in.shape, 'Original Zone',
                    **style['array_in_rect'])
    ax.set_title('Standalone Preprocessed Image', fontsize=FS_TITLE)
    handles.append(mpatches.Patch(**style['array_in_rect'], label='Source Footprint'))

    if array_src_win_read is not None:
        pos, sx, sy = win_to_rect(array_src_win_read, extent)
        ax.add_patch(mpatches.Rectangle(
            pos, sx, sy, **style['array_in_win_read'],
            label='array_src_win_read',
        ))
        handles.append(mpatches.Patch(
            **style['array_in_win_read'], label='Minimal Source Window'))

        pos, sx, sy = win_to_rect(array_src_win_marged, extent)
        ax.add_patch(mpatches.Rectangle(
            pos, sx, sy, **style['array_in_win_marged'],
            label='array_src_win_marged',
        ))
        handles.append(mpatches.Patch(
            **style['array_in_win_marged'],
            label='Required (padded) Support Window'))

    _plot_grid_wire(ax, handles, grid_row, grid_col, grid_mask,
                    grid_mask_valid_value, grid_resolution, win, margin, **style)

    if preprocessed_array_mask_safe_region is not None:
        _plot_array_box(ax, extent, array_in.shape, 'Original Zone',
                        **style['array_in_rect'])

    # ------------------------------------------------------------------
    # Preprocessed mask
    # ------------------------------------------------------------------
    ax = axes[1]
    if preprocessed_array_mask is not None:
        _plot_raster_pixel(ax, preprocessed_array_mask, preprocessed_raster_extent,
                           (nrows_tot, ncols_tot), margin, binary=True, **style)
        ax.set_title('Standalone Preprocessed Mask', fontsize=FS_TITLE)
        _plot_array_box(ax, extent, array_in.shape, 'Original Zone',
                        **style['array_in_rect'])
        handles.extend([
            _legend_separator(),
            _legend_section_title(' - Standalone Preprocessed Mask - '),
            mpatches.Patch(**style['array_in_rect'], label='Source Footprint'),
            ColormapPatch(cmap=cmap_mask, n=2, label='Mask Validity (novalid/valid)'),
        ])

        if preprocessed_array_mask_safe_region is not None:
            safe_region_extent = from_extent(
                preprocessed_array_mask_safe_region, preprocessed_raster_extent,
            )
            safe_region_shape = (
                preprocessed_array_mask_safe_region[:, 1]
                - preprocessed_array_mask_safe_region[:, 0] + 1
            )
            _plot_array_box(ax, safe_region_extent, safe_region_shape,
                            'Safe Region', **style['mask_safe_region'])
            handles.append(mpatches.Patch(
                **style['mask_safe_region'], label='Mask Safe Valid Window'))

        _plot_grid_wire(ax, None, grid_row, grid_col, grid_mask,
                        grid_mask_valid_value, grid_resolution, win, margin, **style)
    else:
        ax.text(0.5, 0.5, 'no mask', ha='center', va='center',
                fontsize=FS_TITLE, color='gray', style='italic')
        ax.axis('off')
    ax.set_title('Preprocessed Mask', fontsize=FS_TITLE)

    # ------------------------------------------------------------------
    # External legend (right of the figure)
    # ------------------------------------------------------------------
    _attach_external_legend(fig, handles)


def _init_plot_resampling_outputs(
    interp, array_in,
    grid_row, grid_col, grid_resolution,
    nodata_out, array_in_origin, win,
    array_in_mask, array_in_mask_safe_win,
    grid_mask, grid_mask_valid_value, grid_nodata,
    array_out_mask, check_boundaries, standalone,
    boundary_condition, trust_padding,
    plot_margin, pixel_size=PIXEL_SIZE, dpi=DPI,
):
    """Run the resampler and prepare the output figure geometry.

    Returns
    -------
    figsize, raster_extent, margin, resampled_array, resampled_mask
        The first three describe the output figure layout; the last two
        are the actual resampling results (the mask may be ``None``).
    """
    resampled_array, resampled_mask = array_grid_resampling(
        interp=interp,
        array_in=array_in,
        grid_row=grid_row,
        grid_col=grid_col,
        grid_resolution=grid_resolution,
        array_out=None,
        array_out_win=None,
        nodata_out=nodata_out,
        array_in_origin=array_in_origin,
        win=win,
        array_in_mask=array_in_mask,
        array_in_mask_safe_win=array_in_mask_safe_win,
        grid_mask=grid_mask,
        grid_mask_valid_value=grid_mask_valid_value,
        grid_nodata=grid_nodata,
        array_out_mask=array_out_mask,
        check_boundaries=check_boundaries,
        standalone=standalone,
        boundary_condition=boundary_condition,
        trust_padding=trust_padding,
        logger_msg_prefix=None,
        logger=None,
    )

    resampled_array = np.atleast_2d(resampled_array)

    margin = np.asarray([[plot_margin, plot_margin], [plot_margin, plot_margin]])
    nrows_tot = resampled_array.shape[0] + np.sum(margin[0])
    ncols_tot = resampled_array.shape[1] + np.sum(margin[1])

    raster_extent = [
        -0.5 + margin[1][0], ncols_tot - margin[1][1] - 0.5,
        nrows_tot - margin[0][1] - 0.5, margin[0][0] - 0.5,
    ]
    figsize = _compute_panel_size(nrows_tot, ncols_tot)
    return figsize, raster_extent, margin, resampled_array, resampled_mask


def _plot_resampling_outputs(
    fig, axes, extent, margin,
    resampled_array, resampled_mask,
    style=PLOT_STYLE,
):
    """Render the *outputs* panel: resampled raster + output mask."""
    nrows, ncols = resampled_array.shape
    nrows_tot = nrows + np.sum(margin[0])
    ncols_tot = ncols + np.sum(margin[1])

    # ------------------------------------------------------------------
    # Resampled raster
    # ------------------------------------------------------------------
    ax = axes[0]
    handles = [_legend_section_title(' - Output Image - ')]
    _plot_raster_pixel(ax, resampled_array, extent,
                       (nrows_tot, ncols_tot), margin, **style)
    _plot_array_box(ax, extent, resampled_array.shape, 'Output Footprint',
                    **style['array_in_rect'])
    ax.set_title('Output Image', fontsize=FS_TITLE)
    handles.append(mpatches.Patch(**style['array_in_rect'], label='Output Footprint'))

    # ------------------------------------------------------------------
    # Optional output mask
    # ------------------------------------------------------------------
    ax = axes[1]
    if resampled_mask is not None:
        _plot_raster_pixel(ax, resampled_mask, extent,
                           (nrows_tot, ncols_tot), margin, binary=True, **style)
        ax.set_title('Output Mask', fontsize=FS_TITLE)
        handles.extend([
            _legend_separator(),
            _legend_section_title(' - Output Mask - '),
            ColormapPatch(cmap=cmap_mask, n=2, label='Mask Validity (novalid/valid)'),
        ])
    else:
        ax.text(0.5, 0.5, 'no mask', ha='center', va='center',
                fontsize=FS_TITLE, color='gray', style='italic')
        ax.set_title('Output Mask', fontsize=FS_TITLE)
        ax.axis('off')

    # ------------------------------------------------------------------
    # External legend (right of the figure)
    # ------------------------------------------------------------------
    _attach_external_legend(fig, handles)


# =============================================================================
# Top-level driver
# =============================================================================

def main_plot(
    interp, array_in,
    grid_row, grid_col, grid_resolution,
    nodata_out, array_in_origin, win,
    array_in_mask, array_in_mask_safe_win,
    grid_mask, grid_mask_valid_value, grid_nodata,
    array_out_mask, check_boundaries, standalone,
    boundary_condition, trust_padding,
    plot_margin=1, pixel_size=PIXEL_SIZE, dpi=DPI,
):
    """Render the three diagnostic figures for a single resampling case.

    The function:

    1. Computes a display margin large enough to host both the grid hull
       excess and the interpolator support pad.
    2. Plots the *Resampling inputs* figure (source raster + input mask).
    3. Plots the *Standalone Preprocessing outputs* figure (padded array
       and mask, plus read / support windows).
    4. Runs :func:`array_grid_resampling` and plots the *Resampling
       outputs* figure (resampled raster + output mask).

    Returns
    -------
    (resampled_array, resampled_mask)
        The values returned by the resampler. The array has ``nodata_out``
        cells already replaced by ``None`` in the displayed copy, but the
        returned ``resampled_array`` keeps the raw ``nodata_out`` values.
    """
    np.set_printoptions(precision=2, suppress=True, linewidth=100)

    # Default win = full resampling output extent.
    if win is None:
        full_shape_out = grid_full_resolution_shape(
            shape=grid_row.shape, resolution=grid_resolution,
        )
        win = np.array(((0, full_shape_out[0] - 1), (0, full_shape_out[1] - 1)))

    # ------------------------------------------------------------------
    # Figure 1: Inputs
    # ------------------------------------------------------------------
    figsize, raster_extent, margin = _plot_resampling_figsize_per_ax(
        interp=interp, array_in=array_in,
        grid_row=np.asarray(grid_row), grid_col=np.asarray(grid_col),
        grid_resolution=grid_resolution,
        nodata_out=nodata_out, array_in_origin=array_in_origin, win=win,
        array_in_mask=array_in_mask, array_in_mask_safe_win=array_in_mask_safe_win,
        grid_mask=grid_mask, grid_mask_valid_value=grid_mask_valid_value,
        grid_nodata=grid_nodata,
        check_boundaries=check_boundaries, standalone=standalone,
        boundary_condition=boundary_condition, trust_padding=trust_padding,
        plot_margin=plot_margin, pixel_size=pixel_size, dpi=dpi,
    )

    fig, axes_ = _build_figure(figsize, 'Resampling inputs')
    _plot_resampling_inputs(
        fig=fig, axes=axes_, extent=raster_extent, margin=margin,
        interp=interp, array_in=array_in,
        grid_row=np.asarray(grid_row), grid_col=np.asarray(grid_col),
        grid_resolution=grid_resolution,
        nodata_out=nodata_out, array_in_origin=array_in_origin, win=win,
        array_in_mask=array_in_mask, array_in_mask_safe_win=array_in_mask_safe_win,
        grid_mask=grid_mask, grid_mask_valid_value=grid_mask_valid_value,
        grid_nodata=grid_nodata,
        check_boundaries=check_boundaries, standalone=standalone,
        boundary_condition=boundary_condition, trust_padding=trust_padding,
        style=PLOT_STYLE,
    )

    # ------------------------------------------------------------------
    # Figure 2: Standalone preprocessing
    # ------------------------------------------------------------------
    fig, axes_ = _build_figure(figsize, 'Standalone Preprocessing outputs')

    # Defensive copy: standalone_preprocessing may mutate its input
    # depending on the interpolator.
    array_in_preprocessing = np.copy(array_in)
    _plot_resampling_preprocessing(
        fig=fig, axes=axes_, extent=raster_extent, margin=margin,
        interp=interp, array_in=array_in_preprocessing,
        grid_row=np.asarray(grid_row), grid_col=np.asarray(grid_col),
        grid_resolution=grid_resolution,
        nodata_out=nodata_out, array_in_origin=array_in_origin, win=win,
        array_in_mask=array_in_mask, array_in_mask_safe_win=array_in_mask_safe_win,
        grid_mask=grid_mask, grid_mask_valid_value=grid_mask_valid_value,
        grid_nodata=grid_nodata,
        check_boundaries=check_boundaries, standalone=standalone,
        boundary_condition=boundary_condition, trust_padding=trust_padding,
        style=PLOT_STYLE,
    )

    # ------------------------------------------------------------------
    # Figure 3: Outputs
    # ------------------------------------------------------------------
    figsize, raster_extent, margin, resampled_array, resampled_mask = (
        _init_plot_resampling_outputs(
            interp=interp, array_in=array_in,
            grid_row=np.asarray(grid_row), grid_col=np.asarray(grid_col),
            grid_resolution=grid_resolution,
            nodata_out=nodata_out, array_in_origin=array_in_origin, win=win,
            array_in_mask=array_in_mask, array_in_mask_safe_win=array_in_mask_safe_win,
            grid_mask=grid_mask, grid_mask_valid_value=grid_mask_valid_value,
            grid_nodata=grid_nodata, array_out_mask=array_out_mask,
            check_boundaries=check_boundaries, standalone=standalone,
            boundary_condition=boundary_condition, trust_padding=trust_padding,
            plot_margin=plot_margin, pixel_size=pixel_size, dpi=dpi,
        )
    )
    # Mask nodata cells for nicer display (NaN -> gray) without altering
    # the returned array.
    resampled_array_ = np.copy(resampled_array).astype(np.float64)
    resampled_array_[np.where(resampled_array == nodata_out)] = np.nan

    fig, axes_ = _build_figure(figsize, 'Resampling outputs')
    _plot_resampling_outputs(
        fig=fig, axes=axes_, extent=raster_extent, margin=margin,
        resampled_array=resampled_array_,
        resampled_mask=resampled_mask,
        style=PLOT_STYLE,
    )

    return resampled_array, resampled_mask


# =============================================================================
# Test-input generators
# =============================================================================

def _mono_point_grid_generate_input(
    row_target,
    col_target,
    array_in_cst_value,
    shape_array_in,
    resolution,
    use_array_in_mask,
    set_array_in_mask_invalid,
    set_array_in_mask_safe_window,
    array_out_mask,
    use_standalone,
    interp,
    nodata_out,
    array_in_origin,
    win,
    use_grid_mask,
    grid_mask_valid_value,
    check_boundaries,
    boundary_condition,
    trust_padding,
    bsplines_kwargs={'epsilon': 1e-2, 'mask_influence_threshold': 1},
):
    """Build a kwargs dict for a single-point (1x1 grid) resampling case.

    The generated grid contains a single node located at
    ``(row_target, col_target)``. This is convenient to probe the resampler
    behaviour pixel by pixel.

    Notes
    -----
    The returned dictionary is meant to be splatted into :func:`main_plot`
    (or directly into :func:`array_grid_resampling`).
    """
    if len(shape_array_in) == 3:
        _, array_in_row, array_in_col = shape_array_in
    else:
        array_in_row, array_in_col = shape_array_in

    kwargs = {}

    # -- interpolator --
    if "bspline" in interp:
        interp = get_interpolator(interp, **bsplines_kwargs)
    else:
        interp = get_interpolator(interp)
    interp.initialize()
    kwargs["interp"] = interp

    # -- input array: either a constant fill or a ramp (np.arange) --
    if array_in_cst_value is not None:
        kwargs['array_in'] = (
            np.ones(np.prod(shape_array_in), dtype=np.float64)
            .reshape(shape_array_in) * array_in_cst_value
        )
    else:
        kwargs['array_in'] = (
            np.arange(np.prod(shape_array_in), dtype=np.float64)
            .reshape(shape_array_in)
        )

    kwargs["array_in_origin"] = array_in_origin
    kwargs['grid_resolution'] = resolution

    # -- input mask --
    kwargs['array_in_mask'] = None
    kwargs['array_in_mask_safe_win'] = None
    if use_array_in_mask:
        array_in_mask = np.ones(kwargs['array_in'].shape, dtype=np.uint8, order='C')
        if set_array_in_mask_invalid:
            # Mark two arbitrary cells as invalid to exercise mask handling.
            array_in_mask[0, 0] = 0
            array_in_mask[17, 27] = 0
        kwargs['array_in_mask'] = array_in_mask
        if set_array_in_mask_safe_window:
            if set_array_in_mask_invalid:
                # Safe window stops one pixel short of the invalid cells.
                kwargs['array_in_mask_safe_win'] = np.asarray(
                    [(1, array_in_row - 1), (1, array_in_col - 1)]
                )
            else:
                # Safe window spans the full array.
                kwargs['array_in_mask_safe_win'] = np.asarray(
                    [(0, array_in_row - 1), (0, array_in_col - 1)]
                )

    # -- single-node grid --
    def _get_scalar_grid(x, y):
        grid_row = np.asarray([[y, ], ])
        grid_col = np.asarray([[x, ], ])
        return np.atleast_2d(grid_row), np.atleast_2d(grid_col)

    grid_row, grid_col = _get_scalar_grid(x=col_target, y=row_target)
    kwargs["grid_row"] = grid_row
    kwargs["grid_col"] = grid_col

    kwargs["array_out_mask"] = array_out_mask

    # -- grid mask --
    grid_mask = None
    if use_grid_mask:
        grid_mask = np.full(np.asarray(grid_row).shape,
                            grid_mask_valid_value, dtype=np.uint8)
        # Force the last column to be invalid (here: the unique node when
        # the grid is 1x1).
        grid_mask[:, -1] = grid_mask_valid_value - 1
    kwargs['grid_mask'] = grid_mask
    kwargs['grid_mask_valid_value'] = grid_mask_valid_value
    kwargs['grid_nodata'] = None

    kwargs["nodata_out"] = nodata_out
    kwargs["win"] = win
    if kwargs["win"] is not None:
        kwargs["win"] = np.asarray(kwargs["win"])

    kwargs["check_boundaries"] = check_boundaries
    kwargs["standalone"] = use_standalone
    kwargs["boundary_condition"] = boundary_condition
    kwargs["trust_padding"] = trust_padding

    return kwargs


def create_grid(
    nrow, ncol, origin_pos, origin_node,
    v_row_y, v_row_x, v_col_y, v_col_x,
    grid_dtype,
):
    """Build an affine-parameterised regular grid of shape (nrow, ncol).

    Given an origin node ``(yo, xo)`` and two basis vectors expressed in
    source-pixel coordinates -- ``(v_row_y, v_row_x)`` for moving along
    the grid's row axis and ``(v_col_y, v_col_x)`` along the grid's column
    axis -- the node at grid position ``(i, j)`` is::

        y[i, j] = yo + (i - origin_pos[1]) * v_row_y + (j - origin_pos[0]) * v_col_y
        x[i, j] = xo + (i - origin_pos[1]) * v_row_x + (j - origin_pos[0]) * v_col_x

    Returns
    -------
    (grid_row, grid_col) : tuple of np.ndarray
        Two arrays of shape ``(nrow, ncol)`` giving each node's row and
        column coordinate in the source raster frame.
    """
    x = np.arange(0, ncol, dtype=grid_dtype)
    y = np.arange(0, nrow, dtype=grid_dtype)
    xx, yy = np.meshgrid(x, y)

    xx -= origin_pos[0]
    yy -= origin_pos[1]

    yyy = origin_node[0] + yy * v_row_y + xx * v_col_y
    xxx = origin_node[1] + yy * v_row_x + xx * v_col_x

    return yyy, xxx


def _multi_points_grid_generate_input(
    array_in_cst_value,
    shape_array_in,
    create_grid_kwargs,
    resolution,
    use_array_in_mask,
    array_in_mask_invalid_slice,
    array_in_mask_safe_window,
    array_out_mask,
    use_standalone,
    interp,
    nodata_out,
    array_in_origin,
    win,
    use_grid_mask,
    grid_mask_valid_value,
    grid_mask_novalid_slice,
    check_boundaries,
    boundary_condition,
    trust_padding,
    bsplines_kwargs={'epsilon': 1e-2, 'mask_influence_threshold': 1},
):
    """Build a kwargs dict for a multi-node grid resampling case.

    Compared to :func:`_mono_point_grid_generate_input`, the grid is now
    described by :func:`create_grid` parameters (``create_grid_kwargs``)
    and both the input mask and the grid mask can be invalidated on
    arbitrary slices via ``array_in_mask_invalid_slice`` and
    ``grid_mask_novalid_slice``.
    """
    if len(shape_array_in) == 3:
        _, array_in_row, array_in_col = shape_array_in
    else:
        array_in_row, array_in_col = shape_array_in

    kwargs = {}

    # -- interpolator --
    if "bspline" in interp:
        interp = get_interpolator(interp, **bsplines_kwargs)
    else:
        interp = get_interpolator(interp)
    interp.initialize()
    kwargs["interp"] = interp

    # -- input array --
    if array_in_cst_value is not None:
        kwargs['array_in'] = (
            np.ones(np.prod(shape_array_in), dtype=np.float64)
            .reshape(shape_array_in) * array_in_cst_value
        )
    else:
        kwargs['array_in'] = (
            np.arange(np.prod(shape_array_in), dtype=np.float64)
            .reshape(shape_array_in)
        )

    kwargs["array_in_origin"] = array_in_origin
    kwargs['grid_resolution'] = resolution

    # -- input mask --
    kwargs['array_in_mask'] = None
    kwargs['array_in_mask_safe_win'] = None
    if use_array_in_mask:
        array_in_mask = np.ones(kwargs['array_in'].shape, dtype=np.uint8, order='C')
        if array_in_mask_invalid_slice is not None:
            array_in_mask[array_in_mask_invalid_slice] = 0
        kwargs['array_in_mask'] = array_in_mask
        if array_in_mask_safe_window:
            kwargs['array_in_mask_safe_win'] = np.asarray(array_in_mask_safe_window)

    # -- grid --
    grid_row, grid_col = create_grid(**create_grid_kwargs)
    kwargs["grid_row"] = grid_row
    kwargs["grid_col"] = grid_col

    kwargs["array_out_mask"] = array_out_mask

    # -- grid mask --
    grid_mask = None
    if use_grid_mask:
        grid_mask = np.full(np.asarray(grid_row).shape,
                            grid_mask_valid_value, dtype=np.uint8)
        if grid_mask_novalid_slice is not None:
            grid_mask[grid_mask_novalid_slice] = grid_mask_valid_value - 1
    kwargs['grid_mask'] = grid_mask
    kwargs['grid_mask_valid_value'] = grid_mask_valid_value
    kwargs['grid_nodata'] = None

    kwargs["nodata_out"] = nodata_out
    kwargs["win"] = win
    if kwargs["win"] is not None:
        kwargs["win"] = np.asarray(kwargs["win"])

    kwargs["check_boundaries"] = check_boundaries
    kwargs["standalone"] = use_standalone
    kwargs["boundary_condition"] = boundary_condition
    kwargs["trust_padding"] = trust_padding

    return kwargs
