"""Shared utilities for the GridR ``chain`` tutorial series.

This module factors out the helpers reused across every
``grid_resampling_chain_*`` notebook so each one stays focused on
demonstrating a single chain feature.

Provided helpers
----------------
- :func:`write_array`            -- write a NumPy array to a GeoTIFF
- :func:`create_grid`            -- build an affine-parameterised regular grid
- :func:`create_star_polygon`    -- build a 6-pointed star Shapely polygon
- :func:`plot_grid_on_image`     -- overlay a grid (with optional masks,
                                    window and geometry polygons) on a
                                    raster image
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import rasterio
import matplotlib.pyplot as plt

from shapely.geometry import Polygon
from shapely.plotting import plot_polygon

from gridr.core.grid.grid_utils import oversample_regular_grid
from notebook_utils import mpl_plot_wrapper


# =============================================================================
# I/O helper
# =============================================================================

def write_array(array: np.ndarray, dtype, fileout: str) -> None:
    """Write a 2D or 3D NumPy array to ``fileout`` as a single-band or
    multi-band GeoTIFF.

    A 3D array of shape ``(bands, height, width)`` is written as a
    multi-band raster; a 2D array of shape ``(height, width)`` is
    written as a single-band raster.
    """
    if array.ndim == 3:
        with rasterio.open(
            fileout, "w", driver="GTiff", dtype=dtype,
            height=array.shape[1], width=array.shape[2],
            count=array.shape[0],
        ) as ds:
            for band in range(array.shape[0]):
                ds.write(array[band].astype(dtype), band + 1)
    elif array.ndim == 2:
        with rasterio.open(
            fileout, "w", driver="GTiff", dtype=dtype,
            height=array.shape[0], width=array.shape[1], count=1,
        ) as ds:
            ds.write(array.astype(dtype), 1)
    else:
        raise ValueError(f"Unsupported array.ndim={array.ndim}")


# =============================================================================
# Grid factory
# =============================================================================

def create_grid(
    nrow: int, ncol: int,
    origin_pos: Tuple[float, float],
    origin_node: Tuple[float, float],
    v_row_y: float, v_row_x: float,
    v_col_y: float, v_col_x: float,
    dtype=np.float64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build an affine-parameterised regular grid of shape ``(nrow, ncol)``.

    Each node ``(i, j)`` is mapped to source-pixel coordinates by::

        y[i, j] = origin_node[0] + (i - origin_pos[1]) * v_row_y
                                  + (j - origin_pos[0]) * v_col_y
        x[i, j] = origin_node[1] + (i - origin_pos[1]) * v_row_x
                                  + (j - origin_pos[0]) * v_col_x

    Returns ``(grid_row, grid_col)`` -- two arrays of shape
    ``(nrow, ncol)`` holding row and column coordinates respectively.
    """
    x = np.arange(0, ncol, dtype=dtype)
    y = np.arange(0, nrow, dtype=dtype)
    xx, yy = np.meshgrid(x, y)
    xx -= origin_pos[0]
    yy -= origin_pos[1]
    grid_row = origin_node[0] + yy * v_row_y + xx * v_col_y
    grid_col = origin_node[1] + yy * v_row_x + xx * v_col_x
    return grid_row, grid_col


# =============================================================================
# Star polygon factory
# =============================================================================

def create_star_polygon(center_x: float, center_y: float, size: float) -> Polygon:
    """Build a 6-pointed star polygon centered at ``(center_x, center_y)``.

    The outer radius equals ``size / 2`` and the inner radius is set so
    the star has the proportions of a regular 6-pointed star.
    """
    outer_radius = size / 2
    inner_radius = outer_radius / math.sqrt(3)
    points = []
    for i in range(12):
        angle_rad = math.radians(90 - i * 30)
        r = outer_radius if i % 2 == 0 else inner_radius
        points.append(
            (center_x + r * math.cos(angle_rad),
             center_y + r * math.sin(angle_rad))
        )
    return Polygon(points)


# =============================================================================
# Diagnostic plot
# =============================================================================

@mpl_plot_wrapper
def plot_grid_on_image(
    z: float,
    grid_row: np.ndarray,
    grid_col: np.ndarray,
    grid_resolution: Tuple[int, int],
    array_shape: Tuple[int, int],
    mask: Optional[np.ndarray] = None,
    win: Optional[np.ndarray] = None,
    raster_image: Optional[np.ndarray] = None,
    raster_image_mask: Optional[np.ndarray] = None,
    geometry_origin: Optional[Tuple[float, float]] = None,
    geometry_pair=None,
    prefix: Optional[str] = None,
):
    """Overlay a resampling grid on a background raster image.

    Parameters
    ----------
    z : float
        Zoom factor for the figure size. The output figure has
        ``z * array_shape`` pixels at 100 dpi.
    grid_row, grid_col : np.ndarray
        Per-node coordinates of the grid in source-raster space.
    grid_resolution : (int, int)
        Oversampling factor of the grid, used only when ``win`` is
        provided (to draw the actual computation window contour).
    array_shape : (int, int)
        Shape of the background raster.
    mask : np.ndarray, optional
        Grid validity mask (``1`` = valid, ``0`` = invalid). When
        provided, nodes are coloured by status (valid / invalid /
        out-of-bounds).
    win : np.ndarray, optional
        Computation window in GridR convention, drawn as an orange
        dashed rectangle.
    raster_image : np.ndarray, optional
        Background image drawn in grey.
    raster_image_mask : np.ndarray, optional
        Source-raster validity mask; invalid pixels are tinted red.
    geometry_origin : (float, float), optional
        Origin shift applied to every geometry in ``geometry_pair``.
    geometry_pair : sequence of two Polygons or None, optional
        ``(valid_polygon, invalid_polygon)``. Either entry may be
        ``None``. Valid polygons are filled green, invalid red.
    prefix : str, optional
        Plot title.
    """
    colors = {
        "blue":   "dodgerblue",
        "red":    "crimson",
        "orange": "darkorange",
        "grey":   "lightsteelblue",
        "green":  "limegreen",
    }
    dpi = 100
    fig = plt.figure(
        figsize=(int(z * array_shape[1]) / dpi,
                 int(z * array_shape[0]) / dpi),
        dpi=dpi,
    )
    ax = fig.add_subplot(111)
    plt.style.use("ggplot")

    if raster_image is not None:
        ax.imshow(
            raster_image, cmap="gray", alpha=0.3,
            extent=[0, array_shape[1], array_shape[0], 0],
        )

    if geometry_origin is not None and geometry_pair is not None:
        oy, ox = geometry_origin
        color_map = [colors["green"], colors["red"]]
        for poly, c in zip(geometry_pair, color_map):
            if poly is not None:
                translated = Polygon(
                    [(p[0] - ox, p[1] - oy) for p in poly.exterior.coords]
                )
                plot_polygon(
                    translated, ax=ax, add_points=False,
                    facecolor=c, edgecolor=c, linewidth=1, alpha=0.3,
                )

    if raster_image_mask is not None:
        invalid_color = (0.863, 0.078, 0.235, 0.7)
        mask_rgba = np.zeros((*raster_image_mask.shape, 4), dtype=np.float32)
        mask_rgba[raster_image_mask == 0] = invalid_color
        ax.imshow(
            mask_rgba,
            extent=[0, array_shape[1], array_shape[0], 0],
            interpolation="nearest",
        )

    if win is not None:
        target_win, _ = oversample_regular_grid(
            grid=np.array((grid_row, grid_col)),
            grid_oversampling_row=grid_resolution[0],
            grid_oversampling_col=grid_resolution[1],
            grid_mask=None, win=win,
        )
        top    = list(zip(target_win[0][0,    :], target_win[1][0,    :]))
        right  = list(zip(target_win[0][:,   -1], target_win[1][:,   -1]))
        bottom = list(zip(target_win[0][-1, ::-1], target_win[1][-1, ::-1]))
        left   = list(zip(target_win[0][::-1, 0], target_win[1][::-1, 0]))
        contour = top + right + bottom + left
        ax.plot(
            [v[1] for v in contour], [v[0] for v in contour],
            linestyle="--", linewidth=2.0, color=colors["orange"],
        )

    # Grid wireframe
    for i in range(grid_row.shape[0]):
        ax.plot(grid_col[i], grid_row[i],
                color=colors["grey"], linewidth=1.5, alpha=0.6)
    for j in range(grid_row.shape[1]):
        ax.plot(grid_col[:, j], grid_row[:, j],
                color=colors["grey"], linewidth=1.5, alpha=0.6)

    # Grid nodes, coloured by status
    if mask is not None:
        masked = np.where(mask == 0)
        oob = np.where(np.logical_or(
            np.logical_or(grid_row < 0., grid_row > array_shape[0] - 1.),
            np.logical_or(grid_col < 0., grid_col > array_shape[1] - 1.),
        ))
        valid = np.where(np.logical_and(
            mask == 1,
            ~np.logical_or(
                np.logical_or(grid_row < 0., grid_row > array_shape[0] - 1.),
                np.logical_or(grid_col < 0., grid_col > array_shape[1] - 1.),
            ),
        ))
        ax.scatter(grid_col[masked].reshape(-1),
                   grid_row[masked].reshape(-1),
                   color=colors["red"], s=z * 8, alpha=1.0,
                   edgecolor="black", linewidth=0.1)
        ax.scatter(grid_col[oob].reshape(-1),
                   grid_row[oob].reshape(-1),
                   color=colors["orange"], s=z * 8, alpha=0.9,
                   edgecolor="black", linewidth=0.1)
        ax.scatter(grid_col[valid].reshape(-1),
                   grid_row[valid].reshape(-1),
                   color=colors["blue"], s=z * 8, alpha=1.0,
                   edgecolor="black", linewidth=0.1)
    else:
        ax.scatter(grid_col.reshape(-1), grid_row.reshape(-1),
                   color=colors["blue"], s=z * 6, alpha=1.0,
                   edgecolor="darkblue", linewidth=0.1)

    ax.set_xlabel("Columns", fontsize=8)
    ax.set_ylabel("Rows", fontsize=8)
    ax.set_xlim(np.min(grid_col) - 10, np.max(grid_col) + 10)
    ax.set_ylim(np.min(grid_row) - 10, np.max(grid_row) + 10)
    if prefix is not None:
        ax.set_title(prefix)
    ax.grid(False)
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    return fig
