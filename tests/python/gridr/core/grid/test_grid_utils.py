# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of GRIDR
# (see https://gitlab.cnes.fr/gridr/gridr).
#
#
"""
Tests for the gridr.core.grid.grid_utils module

Command to run test :
PYTHONPATH=${PWD}/python/:$PYTHONPATH pytest tests/python/gridr/core/grid/test_grid_utils.py
"""
import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

from gridr.core.grid.grid_utils import (
    array_compute_resampling_grid_geometries,
    build_grid,
    interpolate_grid,
    oversample_regular_grid,
)

# Define test data
GRIDx00_r4_c3 = np.array(
    [
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]],
    ]
)
GRIDx00_r4_c3_or5_oc2 = np.array(
    [
        [
            [0.0, 0.5, 1.0, 1.5, 2.0],
            [0.6, 1.1, 1.6, 2.1, 2.6],
            [1.2, 1.7, 2.2, 2.7, 3.2],
            [1.8, 2.3, 2.8, 3.3, 3.8],
            [2.4, 2.9, 3.4, 3.9, 4.4],
            [3.0, 3.5, 4.0, 4.5, 5.0],
            [3.6, 4.1, 4.6, 5.1, 5.6],
            [4.2, 4.7, 5.2, 5.7, 6.2],
            [4.8, 5.3, 5.8, 6.3, 6.8],
            [5.4, 5.9, 6.4, 6.9, 7.4],
            [6.0, 6.5, 7.0, 7.5, 8.0],
            [6.6, 7.1, 7.6, 8.1, 8.6],
            [7.2, 7.7, 8.2, 8.7, 9.2],
            [7.8, 8.3, 8.8, 9.3, 9.8],
            [8.4, 8.9, 9.4, 9.9, 10.4],
            [9.0, 9.5, 10.0, 10.5, 11.0],
        ],
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.1, 0.2, 0.2, 0.2],
            [0.0, 0.2, 0.4, 0.4, 0.4],
            [0.0, 0.3, 0.6, 0.6, 0.6],
            [0.0, 0.4, 0.8, 0.8, 0.8],
            [0.0, 0.5, 1.0, 1.0, 1.0],
            [0.0, 0.5, 1.0, 1.0, 1.0],
            [0.0, 0.5, 1.0, 1.0, 1.0],
            [0.0, 0.5, 1.0, 1.0, 1.0],
            [0.0, 0.5, 1.0, 1.0, 1.0],
            [0.0, 0.5, 1.0, 1.0, 1.0],
        ],
    ]
)

# GRIDMASKx00 : only zeros
# Define expected result for a precision as a tuple (expected, precision)
GRIDMASKx00_r4_c3 = np.zeros(4 * 3, dtype=np.uint8).reshape(4, 3)
GRIDMASKx00_r4_c3_or5_oc2 = (
    np.zeros(((4 - 1) * 5 + 1) * ((3 - 1) * 2 + 1), dtype=np.uint8).reshape(
        ((4 - 1) * 5 + 1), ((3 - 1) * 2 + 1)
    ),
    1e-6,
)

# GRIDMASKx01 : only ones
# Define expected result for a precision as a tuple (expected, precision)
GRIDMASKx01_r4_c3 = np.ones(4 * 3, dtype=np.uint8).reshape(4, 3)
GRIDMASKx01_r4_c3_or5_oc2 = (
    np.ones(((4 - 1) * 5 + 1) * ((3 - 1) * 2 + 1), dtype=np.uint8).reshape(
        ((4 - 1) * 5 + 1), ((3 - 1) * 2 + 1)
    ),
    1e-6,
)

# GRIDMASKx02 :
GRIDMASKx02_r4_c3 = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 1], [0, 1, 1]])
# GRIDMASKx02 : expected nearest :
#  => precision equals to 0.5 (0.5 value will be masked))
GRIDMASKx02_r4_c3_or5_oc2_NN = (
    np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0.0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [0, 1, 1, 1, 1],
        ]
    ),
    0.5,
)
# GRIDMASKx02 : expected strict - non strict zero will be masked
#  => precision equals to 1e-16
GRIDMASKx02_r4_c3_or5_oc2_strict = (
    np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [0, 1, 1, 1, 1],
        ]
    ),
    1e-16,
)


# SECOND_TEST_DATA SET
# Define test data
DTYPE_X100 = np.float32
RESOLUTION_x100 = (7, 4)
SHAPE_x100 = (35, 20)
SHAPE_OUT_x100 = (
    (SHAPE_x100[0] - 1) * RESOLUTION_x100[0] + 1,
    (SHAPE_x100[1] - 1) * RESOLUTION_x100[1] + 1,
)

# Define GRID
Y_x100 = np.linspace(0, (SHAPE_x100[0] - 1) * RESOLUTION_x100[0], SHAPE_x100[0], dtype=DTYPE_X100)
X_x100 = np.linspace(0, (SHAPE_x100[1] - 1) * RESOLUTION_x100[1], SHAPE_x100[1], dtype=DTYPE_X100)
# Target grid coordinates
Y_OUT_x100 = np.linspace(0, (SHAPE_OUT_x100[0] - 1), SHAPE_OUT_x100[0], dtype=DTYPE_X100)
X_OUT_x100 = np.linspace(0, (SHAPE_OUT_x100[1] - 1), SHAPE_OUT_x100[1], dtype=DTYPE_X100)


GRID_IN_ARRAY_ROW_x100 = np.arange(np.prod(SHAPE_x100), dtype=np.float32).reshape(SHAPE_x100)
GRID_IN_ARRAY_COL_x100 = 10 + 10.0 * np.arange(np.prod(SHAPE_x100), dtype=np.float32).reshape(
    SHAPE_x100
)
GRID_IN_ARRAY_x100 = np.stack((GRID_IN_ARRAY_ROW_x100, GRID_IN_ARRAY_COL_x100))
# Compute expected output

# Create the "sparse" coordinates grid in order to preserve memory
x_new_sparse_x100, y_new_sparse_x100 = np.meshgrid(
    X_OUT_x100, Y_OUT_x100, indexing="xy", sparse=True
)
# rows
GRID_OUT_ARRAY_ROW_x100 = np.empty((SHAPE_OUT_x100[0], SHAPE_OUT_x100[1]), dtype=DTYPE_X100)
GRID_OUT_ARRAY_COL_x100 = np.empty((SHAPE_OUT_x100[0], SHAPE_OUT_x100[1]), dtype=DTYPE_X100)
interpolator_row_x100 = RegularGridInterpolator(
    (Y_x100, X_x100), GRID_IN_ARRAY_ROW_x100, method="linear", bounds_error=False, fill_value=np.nan
)
interpolator_col_x100 = RegularGridInterpolator(
    (Y_x100, X_x100), GRID_IN_ARRAY_COL_x100, method="linear", bounds_error=False, fill_value=np.nan
)
GRID_OUT_ARRAY_ROW_x100[:, :] = interpolator_row_x100((y_new_sparse_x100, x_new_sparse_x100))
GRID_OUT_ARRAY_COL_x100[:, :] = interpolator_col_x100((y_new_sparse_x100, x_new_sparse_x100))
GRID_OUT_ARRAY_x100 = np.stack((GRID_OUT_ARRAY_ROW_x100, GRID_OUT_ARRAY_COL_x100))
assert np.all(
    GRID_OUT_ARRAY_x100[:, :: RESOLUTION_x100[0], :: RESOLUTION_x100[1]].shape
    == GRID_IN_ARRAY_x100.shape
)
np.testing.assert_array_equal(
    GRID_OUT_ARRAY_x100[:, :: RESOLUTION_x100[0], :: RESOLUTION_x100[1]], GRID_IN_ARRAY_x100
)

# Test data for array_compute_resampling_grid_geometries
GRIDx03_GEN_ROW_VEC = np.array((0.0, 1.0))
GRIDx03_GEN_COL_VEC = np.array((3.0, 0.0))
GRIDx03_GEN_ORIGIN = np.array((100.0, 200.0))
# Create index grids
GRIDx03_ROWS_r4_c3, GRIDx03_COLS_r4_c3 = np.meshgrid(
    np.arange(4, dtype=np.float64), np.arange(3, dtype=np.float64), indexing="ij"
)
GRIDx03_r4_c3 = np.array(
    (
        GRIDx03_GEN_ORIGIN[1]
        + GRIDx03_ROWS_r4_c3 * GRIDx03_GEN_ROW_VEC[1]
        + GRIDx03_COLS_r4_c3 * GRIDx03_GEN_COL_VEC[1],
        GRIDx03_GEN_ORIGIN[0]
        + GRIDx03_ROWS_r4_c3 * GRIDx03_GEN_ROW_VEC[0]
        + GRIDx03_COLS_r4_c3 * GRIDx03_GEN_COL_VEC[0],
    )
)
GRIDMASKx03_r4_c3_FULL_ZEROS = np.zeros(GRIDx03_r4_c3[0].shape, dtype=np.uint8)
GRIDMASKx03_r4_c3_FULL_ONES = np.ones(GRIDx03_r4_c3[0].shape, dtype=np.uint8)
GRIDMASKx03_r4_c3_ONE_ONE = np.zeros(GRIDx03_r4_c3[0].shape, dtype=np.uint8)
GRIDMASKx03_r4_c3_ONE_ONE[1, 1] = 1


class TestGridUtils:
    """Test class"""

    @pytest.mark.parametrize(
        "data, expected, testing_decimal",
        [
            (
                (GRIDx03_r4_c3, (1, 1), None, None, None, None),
                (object, GRIDx03_GEN_COL_VEC, GRIDx03_GEN_ROW_VEC),
                6,
            ),
            (
                (GRIDx03_r4_c3, (2, 3), None, None, None, None),
                (object, GRIDx03_GEN_COL_VEC / 3.0, GRIDx03_GEN_ROW_VEC / 2.0),
                6,
            ),
            (
                (GRIDx03_r4_c3, (1, 1), GRIDMASKx03_r4_c3_FULL_ZEROS, 1, None, None),
                (None, None, None),
                6,
            ),
            (
                (GRIDx03_r4_c3, (1, 1), GRIDMASKx03_r4_c3_FULL_ONES, 1, None, None),
                (object, GRIDx03_GEN_COL_VEC, GRIDx03_GEN_ROW_VEC),
                6,
            ),
            (
                (GRIDx03_r4_c3, (1, 1), GRIDMASKx03_r4_c3_ONE_ONE, 1, None, None),
                (object, None, None),
                6,
            ),
            (
                (GRIDx03_r4_c3, (1, 1), GRIDMASKx03_r4_c3_FULL_ONES, 1, -999.0, None),
                (Exception, None, None),
                6,
            ),  # excluse mask parameters
            (
                (
                    GRIDx03_r4_c3,
                    (1, 1),
                    GRIDMASKx03_r4_c3_FULL_ONES.astype(np.float32),
                    1,
                    None,
                    None,
                ),
                (Exception, None, None),
                6,
            ),  # mask type
            (
                (GRIDx03_r4_c3.astype(np.float32), (1, 1), None, 1, None, None),
                (Exception, None, None),
                6,
            ),  # grid type
            (
                (GRIDx03_r4_c3, (1, 1), None, 1, None, np.array(((0, 3), (0, 2)))),
                (object, GRIDx03_GEN_COL_VEC, GRIDx03_GEN_ROW_VEC),
                6,
            ),  # check the full window
            (
                (GRIDx03_r4_c3, (1, 1), None, 1, None, np.array(((0, 4), (0, 2)))),
                (ValueError, None, None),
                6,
            ),  # check out of bounds window
            (
                (GRIDx03_r4_c3, (1, 1), None, 1, None, np.array(((0, 0), (0, 0)))),
                (object, None, None),
                6,
            ),  # check empty window => w1 and w2 cannot be computed
        ],
    )
    def test_array_compute_resampling_grid_geometries(self, data, expected, testing_decimal):
        """Test a regular grid interpolation with mask

        Args:
            data : input data as a tuple (grid, resolution, mask, mask_value, mask_nodata, window)
            expected: expected data as a tuple containing :
                    - expected_grid_metrics indicator : None or True or Exception
                    - the expected w1 vector
                    - the expected w2 vector
            testing_decimal: decimal precision used for np.testing.assert_array_almost_equal
        """
        # data : (grid, resolution, mask, mask_value, mask_nodata, window)
        # expected : (grid_metrics (object or None), transition_matrix_w1, transition_matrix_w2)
        grid, resolution, grid_mask, grid_mask_valid_value, grid_nodata, window = data
        expected_grid_metrics, expected_w1, expected_w2 = expected

        try:
            grid_metrics = array_compute_resampling_grid_geometries(
                grid_row=grid[0],
                grid_col=grid[1],
                grid_resolution=resolution,
                win=window,
                grid_mask=grid_mask,
                grid_mask_valid_value=grid_mask_valid_value,
                grid_nodata=grid_nodata,
            )
        except Exception as e:
            try:
                if issubclass(expected_grid_metrics, Exception):
                    # its ok but do not go further
                    return
                else:
                    raise e
            except TypeError:
                raise e
        else:
            try:
                if issubclass(expected_grid_metrics, Exception):
                    raise Exception("Should have raised an exception")
            except TypeError:
                pass

        # Check
        if grid_metrics is not None and expected_grid_metrics is None:
            raise Exception("Grid metrics must have be None")
        if grid_metrics is None and expected_grid_metrics is not None:
            raise Exception("Grid metrics must not be None")
        else:
            return

        if expected_w1 is not None:
            np.testing.assert_array_almost_equal(
                expected_w1, grid_metrics.transition_matrix.w1, decimal=testing_decimal
            )
        else:
            assert grid_metrics.transition_matrix.w1 is None

        if expected_w2 is not None:
            np.testing.assert_array_almost_equal(
                expected_w2, grid_metrics.transition_matrix.w2, decimal=testing_decimal
            )
        else:
            assert grid_metrics.transition_matrix.w2 is None

    @pytest.mark.parametrize(
        "data, expected, testing_decimal",
        [
            ((GRIDx00_r4_c3, None), ((5, 2), GRIDx00_r4_c3_or5_oc2, (None, 0)), 6),
            (
                (GRIDx00_r4_c3, GRIDMASKx00_r4_c3),
                ((5, 2), GRIDx00_r4_c3_or5_oc2, GRIDMASKx00_r4_c3_or5_oc2),
                6,
            ),
            (
                (GRIDx00_r4_c3, GRIDMASKx01_r4_c3),
                ((5, 2), GRIDx00_r4_c3_or5_oc2, GRIDMASKx01_r4_c3_or5_oc2),
                6,
            ),
            (
                (GRIDx00_r4_c3, GRIDMASKx02_r4_c3),
                ((5, 2), GRIDx00_r4_c3_or5_oc2, GRIDMASKx02_r4_c3_or5_oc2_NN),
                6,
            ),
            (
                (GRIDx00_r4_c3, GRIDMASKx02_r4_c3),
                ((5, 2), GRIDx00_r4_c3_or5_oc2, GRIDMASKx02_r4_c3_or5_oc2_strict),
                6,
            ),
        ],
    )
    def test_interpolate_grid(self, data, expected, testing_decimal):
        """Test a regular grid interpolation with mask

        Args:
            data : input data as a tuple containing the grid and the mask_binarize_precision
            expected: expected data as a tuple containing :
                    - the oversampling tuple (oversampling_y, oversampling_x)
                    - the expected interpolated grid
                    - a tuple containing (expected_grid_mask, mask_precision)
            testing_decimal: decimal precision used for np.testing.assert_array_almost_equal
        """
        # Create intial coordinates so that new coordinates match an integer sequence
        grid, grid_mask = data
        (oversampling_y, oversampling_x), expected_grid, (expected_grid_mask, mask_precision) = (
            expected
        )
        _, nrow, ncol = grid.shape

        x = np.arange(ncol) * oversampling_x
        y = np.arange(nrow) * oversampling_y
        x_new = np.linspace(x[0], x[-1], num=(ncol - 1) * oversampling_x + 1, endpoint=True)
        assert x_new[0] == x[0]
        assert x_new[-1] == x[-1]
        np.testing.assert_almost_equal(x_new[1], x[0] + 1)

        y_new = np.linspace(y[0], y[-1], num=(nrow - 1) * oversampling_y + 1, endpoint=True)
        assert y_new[0] == y[0]
        assert y_new[-1] == y[-1]
        np.testing.assert_almost_equal(y_new[1], y[0] + 1)

        interp_grid, interp_grid_mask = interpolate_grid(
            grid=grid,
            grid_mask=grid_mask,
            x=x,
            y=y,
            x_new=x_new,
            y_new=y_new,
            mask_binarize_precision=mask_precision,
        )

        # Check
        np.testing.assert_array_equal(interp_grid.shape, expected_grid.shape)
        np.testing.assert_array_almost_equal(interp_grid, expected_grid, decimal=testing_decimal)
        if grid_mask is None:
            assert expected_grid_mask is None
        else:
            np.testing.assert_array_almost_equal(
                interp_grid_mask, expected_grid_mask, decimal=testing_decimal
            )

    @pytest.mark.parametrize(
        "data, expected, testing_decimal",
        [
            ((GRIDx00_r4_c3, None), ((5, 2), GRIDx00_r4_c3_or5_oc2, (None, 0)), 6),
            (
                (GRIDx00_r4_c3, GRIDMASKx00_r4_c3),
                ((5, 2), GRIDx00_r4_c3_or5_oc2, GRIDMASKx00_r4_c3_or5_oc2),
                6,
            ),
            (
                (GRIDx00_r4_c3, GRIDMASKx01_r4_c3),
                ((5, 2), GRIDx00_r4_c3_or5_oc2, GRIDMASKx01_r4_c3_or5_oc2),
                6,
            ),
            (
                (GRIDx00_r4_c3, GRIDMASKx02_r4_c3),
                ((5, 2), GRIDx00_r4_c3_or5_oc2, GRIDMASKx02_r4_c3_or5_oc2_NN),
                6,
            ),
            (
                (GRIDx00_r4_c3, GRIDMASKx02_r4_c3),
                ((5, 2), GRIDx00_r4_c3_or5_oc2, GRIDMASKx02_r4_c3_or5_oc2_strict),
                6,
            ),
        ],
    )
    @pytest.mark.parametrize(
        "window",
        (
            (None, True),
            (np.array([[4, 7], [2, 3]]), True),
            (np.array([[4, 7], [2, 6]]), False),  # Outside of domain exception
        ),
    )
    def test_oversample_regular_grid(self, data, expected, window, testing_decimal):
        """Test the get_oversampled_grid method

        Args:
            data : input data as a tuple containing the grid and the mask_binarize_precision
            expected: expected data as a tuple containing :
                    - the oversampling tuple (oversampling_y, oversampling_x)
                    - the expected interpolated grid
                    - a tuple containing (expected_grid_mask, mask_precision)
            window: computing window
            testing_decimal: decimal precision used for np.testing.assert_array_almost_equal
        """
        # Create intial coordinates so that new coordinates match an integer sequence
        grid, grid_mask = data
        (oversampling_y, oversampling_x), expected_grid, (expected_grid_mask, mask_precision) = (
            expected
        )
        _, nrow, ncol = grid.shape
        window, window_ok = window
        if window is not None:
            expected_grid = expected_grid[
                :, window[0, 0] : window[0, 1] + 1, window[1, 0] : window[1, 1] + 1
            ]
            if expected_grid_mask is not None:
                expected_grid_mask = expected_grid_mask[
                    window[0, 0] : window[0, 1] + 1, window[1, 0] : window[1, 1] + 1
                ]

        try:
            oversampled_grid, oversampled_grid_mask = oversample_regular_grid(
                grid=grid,
                grid_oversampling_row=oversampling_y,
                grid_oversampling_col=oversampling_x,
                grid_mask=grid_mask,
                grid_mask_binarize_precision=mask_precision,
                win=window,
            )
        except Exception as e:
            if not window_ok:
                # its ok but do not go further
                return
            else:
                raise e
        else:
            if not window_ok:
                raise Exception("Should have raised an outside of domain exception")

        # Check
        np.testing.assert_array_equal(oversampled_grid.shape, expected_grid.shape)
        np.testing.assert_array_almost_equal(
            oversampled_grid, expected_grid, decimal=testing_decimal
        )
        if grid_mask is None:
            assert expected_grid_mask is None
        else:
            np.testing.assert_array_almost_equal(
                oversampled_grid_mask, expected_grid_mask, decimal=testing_decimal
            )

    @pytest.mark.parametrize(
        "data, expected, testing_decimal",
        [
            (
                (GRIDx00_r4_c3),
                (
                    (5, 2),
                    (1, 1),
                    GRIDx00_r4_c3_or5_oc2,
                ),
                6,
            ),
            (
                (GRID_IN_ARRAY_x100),
                (
                    RESOLUTION_x100,
                    (1, 1),
                    GRID_OUT_ARRAY_x100,
                ),
                4,
            ),
            # ((GRIDx00_r4_c3, GRIDMASKx00_r4_c3), ((5, 2), GRIDx00_r4_c3_or5_oc2,
            #     GRIDMASKx00_r4_c3_or5_oc2), 6),
            # ((GRIDx00_r4_c3, GRIDMASKx01_r4_c3), ((5, 2), GRIDx00_r4_c3_or5_oc2,
            #     GRIDMASKx01_r4_c3_or5_oc2), 6),
            # ((GRIDx00_r4_c3, GRIDMASKx02_r4_c3), ((5, 2), GRIDx00_r4_c3_or5_oc2,
            #    GRIDMASKx02_r4_c3_or5_oc2_NN), 6),
            # ((GRIDx00_r4_c3, GRIDMASKx02_r4_c3), ((5, 2), GRIDx00_r4_c3_or5_oc2,
            #    GRIDMASKx02_r4_c3_or5_oc2_strict), 6),
        ],
    )
    @pytest.mark.parametrize(
        "window",
        (
            (None, True),
            # (np.array([[4,7],[2,3]]), True),
            # (np.array([[4,7],[2,6]]), False), # Outside of domain exception
        ),
    )
    def test_build_grid(self, data, expected, window, testing_decimal):
        """Test the get_oversampled_grid method

        Args:
            data : input data as a tuple containing the grid and the mask_binarize_precision
            expected: expected data as a tuple containing :
                    - the oversampling tuple (oversampling_y, oversampling_x)
                    - the expected interpolated grid
                    - a tuple containing (expected_grid_mask, mask_precision)
            window: computing window
            testing_decimal: decimal precision used for np.testing.assert_array_almost_equal
        """
        # Create intial coordinates so that new coordinates match an integer sequence
        grid = data
        grid_resolution, out_resolution, expected_grid = expected
        _, nrow, ncol = grid.shape
        window, window_ok = window

        if window is not None:
            expected_grid = expected_grid[
                :, window[0, 0] : window[0, 1] + 1, window[1, 0] : window[1, 1] + 1
            ]

        try:
            oversampled_grid = build_grid(
                resolution=out_resolution,
                grid=grid,
                grid_target_win=window,
                grid_resolution=grid_resolution,
                out=None,
                computation_dtype=grid.dtype,
            )

        except Exception as e:
            if not window_ok:
                # its ok but do not go further
                return
            else:
                raise e
        else:
            if not window_ok:
                raise Exception("Should have raised an outside of domain exception")

        # Check
        np.testing.assert_array_equal(oversampled_grid.shape, expected_grid.shape)
        np.testing.assert_array_almost_equal(
            oversampled_grid, expected_grid, decimal=testing_decimal
        )
