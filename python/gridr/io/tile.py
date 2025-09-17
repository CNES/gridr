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
Grid commons module
"""
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window

from gridr.core.utils.array_utils import ArrayProfile
from gridr.core.utils.array_window import from_rio_window, window_check
from gridr.core.utils.chunks import get_chunk_boundaries


def read_tile_edges(
    ds: rasterio.io.DatasetReader,
    ds_band: int,
    tile_shape: Tuple[int, int],
    merge: bool = False,
    window: Optional[Window] = None,
    check: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts the row and column edges of tiles within a specified window of a
    raster dataset.

    This function identifies the boundary rows and columns of tiles (as defined
    by `tile_shape`) within a given window of a raster and reads the
    corresponding pixel values along those edges.

    Parameters
    ----------
    ds : rasterio.io.DatasetReader
        The open raster dataset to read from.

    ds_band : int
        The band index (1-based) to read from the dataset.

    tile_shape : Tuple[int, int]
        The shape of the tiles as (number of rows, number of columns) used to
        determine tile boundaries.

    merge : bool, optional
        If True, tiles that are smaller than `tile_shape` at the edge are merged
        with the adjacent tile along the corresponding dimension.
        Defaults to False.

    window : Optional[Window], optional
        A rasterio `Window` defining the region of the dataset to consider. The
        top-left corner of this window serves as the origin for tile alignment.
        If None, the full extent of the dataset is used.
        Defaults to None.

    check : bool, optional
        If True, checks the consistency of the provided window against the
        dataset shape. Raises `ValueError` if the check fails.
        Defaults to True.

    Returns
    -------
    edge_row_indices : np.ndarray
        1D array of global row indices corresponding to the horizontal edges of
        the tiles.

    edge_col_indices : np.ndarray
        1D array of global column indices corresponding to the vertical edges of
        the tiles.

    edge_rows : np.ndarray
        2D array containing the pixel values along each horizontal tile edge.
        Each row corresponds to a horizontal edge line.

    edge_cols : np.ndarray
        2D array containing the pixel values along each vertical tile edge.
        Each row corresponds to a vertical edge line.

    Raises
    ------
    ValueError
        If `check` is True and the provided window is not valid relative to the
        dataset.

    Notes
    -----

    -   The `edge_rows` array has shape (number of tile row edges, width of the
        window).
    -   The `edge_cols` array has shape (number of tile column edges, height of
        the window).
    -   The output arrays are allocated with `order='C'` for compatibility with
        downstream processing.

    """
    # Default window to full dataset
    if window is None:
        window = Window(0, 0, ds.width, ds.height)
    elif check:
        arr = ArrayProfile.from_dataset(ds)
        arr_win = from_rio_window(window)
        if not window_check(arr, arr_win, axes=None):
            raise ValueError(
                f"window check fails : check window {arr_win} / " f"array {arr.shape} consistency"
            )

    row_start = int(window.row_off)
    col_start = int(window.col_off)
    # row_stop = row_start + int(window.height)
    # col_stop = col_start + int(window.width)

    # Get chunks considering the set merge policy.
    row_chunks = get_chunk_boundaries(window.height, tile_shape[0], merge)
    col_chunks = get_chunk_boundaries(window.width, tile_shape[1], merge)

    # The chunks returns intervals so we have to take unique only unique borders
    # and apply the window origin
    # Adjust the last chunk boundary to ensure exclusivity of the right edge.
    # Since chunk boundaries are intended to be used as slice intervals (i.e.,
    # [start:end]), the last boundary must be decreased by one to maintain
    # consistency.
    row_chunks[-1, 1] -= 1
    col_chunks[-1, 1] -= 1
    edge_row_indices = row_start + np.unique(row_chunks)
    edge_col_indices = col_start + np.unique(col_chunks)

    # Allocate C-contiguous buffers to store the edges
    edge_rows = np.empty(
        (edge_row_indices.size, window.width), dtype=ds.dtypes[ds_band - 1], order="C"
    )
    edge_cols = np.empty(
        (edge_col_indices.size, window.height), dtype=ds.dtypes[ds_band - 1], order="C"
    )

    for idx, row_idx in enumerate(edge_row_indices):
        # Set the current window to read the row at `row_idx` considering the
        # column windowing
        cwin = Window(col_start, row_idx, window.width, 1)
        data = ds.read(ds_band, window=cwin)
        edge_rows[idx, :] = data.reshape(1, -1)[:]

    for idx, col_idx in enumerate(edge_col_indices):
        # Set the current window to read the column at `col_idx` considering the
        # row windowing
        cwin = Window(col_idx, row_start, 1, window.height)
        data = ds.read(ds_band, window=cwin)
        edge_cols[idx, :] = data.reshape(1, -1)[:]

    return edge_row_indices, edge_col_indices, edge_rows, edge_cols
