# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of GRIDR
# (see https://gitlab.cnes.fr/gridr/gridr).
#
#
"""
Tests for the gridr.core.io.tile module

Command to run test :
PYTHONPATH=${PWD}/python/:$PYTHONPATH pytest tests/python/gridr/io/test_tile.py
"""
import numpy as np
import pytest
from rasterio.io import MemoryFile
from rasterio.windows import Window

from gridr.io.tile import read_tile_edges


class TestReadTileEdges:
    @pytest.fixture(
        autouse=True,
        params=[
            {"shape": (100, 120), "desc": "multiple"},  # Multiple of tile size (20, 30)
            {"shape": (103, 117), "desc": "non-multiple"},  # Not a multiple
        ],
    )
    def dataset(self, request):
        """
        Fixture creating an in-memory dataset with known pixel values
        and shape either matching or not matching tile multiples.
        """
        shape = request.param["shape"]
        data = np.arange(shape[0] * shape[1], dtype=np.uint16).reshape((1, *shape))
        profile = {
            "driver": "GTiff",
            "height": shape[0],
            "width": shape[1],
            "count": 1,
            "dtype": "uint16",
        }

        with MemoryFile() as memfile:
            with memfile.open(**profile) as ds:
                ds.write(data)
            with memfile.open() as ds:
                self.ds = ds
                self.data = data[0]
                self.shape = shape
                yield  # The dataset and associated info are available as class attributes

    @pytest.mark.parametrize("merge", [False, True])
    @pytest.mark.parametrize("check", [False, True])
    @pytest.mark.parametrize("window", [None, Window(3, 5, 90, 80)])
    def test_combinations(self, merge, check, window):
        """
        Test all parameter combinations for `read_tile_edges` with:
        - merge: whether to merge partial tiles
        - check: whether to validate window consistency
        - window: None (full dataset) or a defined subwindow

        This test covers both dataset shapes: multiple and non-multiple of tile size.
        """
        tile_shape = (20, 30)
        ds = self.ds
        data = self.data

        # Use full dataset window if no subwindow is specified
        win = window or Window(0, 0, ds.width, ds.height)
        row_start, col_start = int(win.row_off), int(win.col_off)
        row_stop = row_start + int(win.height)
        col_stop = col_start + int(win.width)

        # Call the function under test
        rows_idx, cols_idx, edge_rows, edge_cols = read_tile_edges(
            ds=ds, ds_band=1, tile_shape=tile_shape, merge=merge, window=window, check=check
        )

        # Compute the expected edge indices for rows or columns given a tile size.
        #
        # - Edges are computed as the boundaries between tiles, using a regular step.
        # - If the image size is not a multiple of the tile size, the last pixel
        #   index (stop - 1) is always included to ensure full coverage of the image.
        # - When `merge` is True, and the last tile is smaller than the expected
        #   tile size, it is merged with the previous one by shifting the last edge
        #   to replace the penultimate one.
        # - Finally, the edges are deduplicated and sorted.
        def expected_edges(start, stop, step, merge):
            edges = list(range(0, stop - start, step))
            last_edge = (stop - start) - 1
            if last_edge not in edges:
                edges.append(last_edge)
            if merge and len(edges) >= 2 and (edges[-1] - edges[-2] < step - 1):
                edges[-2] = edges[-1]
            return sorted(set(edges))

        expected_row_idx = np.array(
            [row_start + i for i in expected_edges(row_start, row_stop, tile_shape[0], merge)]
        )
        expected_col_idx = np.array(
            [col_start + i for i in expected_edges(col_start, col_stop, tile_shape[1], merge)]
        )

        # Check that the returned indices match the expected ones
        np.testing.assert_array_equal(rows_idx, expected_row_idx)
        np.testing.assert_array_equal(cols_idx, expected_col_idx)

        # Check output array dimensions
        assert edge_rows.shape == (len(expected_row_idx), int(win.width))
        assert edge_cols.shape == (len(expected_col_idx), int(win.height))

        # Check row data
        for i, r in enumerate(expected_row_idx):
            expected_row = data[r, col_start:col_stop]
            np.testing.assert_array_equal(edge_rows[i], expected_row)

        # Check column data
        for i, c in enumerate(expected_col_idx):
            expected_col = data[row_start:row_stop, c]
            np.testing.assert_array_equal(edge_cols[i], expected_col)

    def test_window_larger_than_dataset(self):
        """
        Ensure that when the requested window exceeds the dataset bounds:
        - An error is raised if check=True
        - The function handles it gracefully if check=False
        """
        ds = self.ds
        shape = self.shape
        # data = self.data
        tile_shape = (20, 30)

        # Create an oversized window (goes beyond dataset width/height)
        oversized_window = Window(0, 0, shape[1] + 50, shape[0] + 50)

        # Case 1: check=True should raise a ValueError
        with pytest.raises(ValueError, match="window check fails"):
            read_tile_edges(
                ds=ds,
                ds_band=1,
                tile_shape=tile_shape,
                window=oversized_window,
                check=True,
            )

        # Case 2: check=False should raise an error
        with pytest.raises(ValueError):
            row_idx, col_idx, edge_rows, edge_cols = read_tile_edges(
                ds=ds,
                ds_band=1,
                tile_shape=tile_shape,
                window=oversized_window,
                check=False,
            )
