# coding: utf8
#
# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of GRIDR
# (see https://gitlab.cnes.fr/gridr/gridr).
#
#
"""
Tests for the gridr.core.grid.grid_mask module

Command to run test :
PYTHONPATH=${PWD}/python/:$PYTHONPATH pytest tests/python/gridr/core/grid/test_grid_resampling.py
"""
import numpy as np
import pytest

from gridr.core.grid.grid_mask import Validity
from gridr.core.grid.grid_resampling import array_grid_resampling
from gridr.core.interp.interpolator import get_interpolator

UNMASKED_VALUE = Validity.VALID
MASKED_VALUE = Validity.INVALID


def create_grid(
    nrow, ncol, origin_pos, origin_node, v_row_y, v_row_x, v_col_y, v_col_x, grid_dtype
):
    """ """
    x = np.arange(0, ncol, dtype=grid_dtype)
    y = np.arange(0, nrow, dtype=grid_dtype)
    xx, yy = np.meshgrid(x, y)

    xx -= origin_pos[0]
    yy -= origin_pos[1]

    yyy = origin_node[0] + yy * v_row_y + xx * v_col_y
    xxx = origin_node[1] + yy * v_row_x + xx * v_col_x

    return yyy, xxx


ARRAY_IN_001_SHAPE = (4, 5)
ARRAY_IN_001_DTYPE = np.float64
ARRAY_IN_001_ARRAY = np.arange(np.prod(ARRAY_IN_001_SHAPE), dtype=ARRAY_IN_001_DTYPE).reshape(
    ARRAY_IN_001_SHAPE
)
MASK_IN_001_DTYPE = np.uint8
# Full valid
MASK_IN_001_ARRAY_01 = np.full(ARRAY_IN_001_SHAPE, UNMASKED_VALUE, dtype=MASK_IN_001_DTYPE)
# Full invalid
MASK_IN_001_ARRAY_02 = np.full(ARRAY_IN_001_SHAPE, MASKED_VALUE, dtype=MASK_IN_001_DTYPE)

# First grid is idendity
GRID_IN_001_01_SHAPE = ARRAY_IN_001_SHAPE
GRID_IN_001_01_DTYPE = np.float64
# create idendity grid - row first
GRID_IN_001_01_GRID = np.meshgrid(
    np.arange(ARRAY_IN_001_SHAPE[0], dtype=GRID_IN_001_01_DTYPE),
    np.arange(ARRAY_IN_001_SHAPE[1], dtype=GRID_IN_001_01_DTYPE),
    indexing="ij",
)


class TestGridResampling:
    """Test class"""

    @pytest.mark.parametrize(
        "array, grid, interp, mask, grid_mask, grid_mask_valid_value, grid_nodata, "
        "check_boundaries, testing_decimal",
        [
            (ARRAY_IN_001_ARRAY, GRID_IN_001_01_GRID, "nearest", None, None, None, None, True, 6),
            (
                ARRAY_IN_001_ARRAY,
                GRID_IN_001_01_GRID,
                "nearest",
                MASK_IN_001_ARRAY_01,
                None,
                None,
                None,
                True,
                6,
            ),
            (
                ARRAY_IN_001_ARRAY,
                GRID_IN_001_01_GRID,
                "nearest",
                MASK_IN_001_ARRAY_02,
                None,
                None,
                None,
                True,
                6,
            ),
            (
                ARRAY_IN_001_ARRAY,
                GRID_IN_001_01_GRID,
                "nearest",
                None,
                None,
                None,
                None,
                False,
                6,
            ),  # with nearest idendity we can deactivate boundaries check
            (ARRAY_IN_001_ARRAY, GRID_IN_001_01_GRID, "linear", None, None, None, None, True, 6),
            (
                ARRAY_IN_001_ARRAY,
                GRID_IN_001_01_GRID,
                "linear",
                MASK_IN_001_ARRAY_01,
                None,
                None,
                None,
                True,
                6,
            ),
            (
                ARRAY_IN_001_ARRAY,
                GRID_IN_001_01_GRID,
                "linear",
                MASK_IN_001_ARRAY_02,
                None,
                None,
                None,
                True,
                6,
            ),
            (ARRAY_IN_001_ARRAY, GRID_IN_001_01_GRID, "cubic", None, None, None, None, True, 6),
            (
                ARRAY_IN_001_ARRAY,
                GRID_IN_001_01_GRID,
                "cubic",
                MASK_IN_001_ARRAY_01,
                None,
                None,
                None,
                True,
                6,
            ),
            (
                ARRAY_IN_001_ARRAY,
                GRID_IN_001_01_GRID,
                "cubic",
                MASK_IN_001_ARRAY_02,
                None,
                None,
                None,
                True,
                6,
            ),
            (
                np.asarray([ARRAY_IN_001_ARRAY, ARRAY_IN_001_ARRAY]),
                GRID_IN_001_01_GRID,
                "cubic",
                MASK_IN_001_ARRAY_02,
                None,
                None,
                None,
                True,
                6,
            ),
        ],
    )
    def test_grid_resampling_standalone_no_idendity_1_1(
        self,
        array,
        grid,
        interp,
        mask,
        grid_mask,
        grid_mask_valid_value,
        grid_nodata,
        check_boundaries,
        testing_decimal,
    ):
        """Test idendity resampling with a full resolute grid"""
        nodata_out = -10
        standalone = False
        boundary_condition = None

        array_out, mask_out = array_grid_resampling(
            interp=interp,
            array_in=array,
            grid_row=grid[0],
            grid_col=grid[1],
            grid_resolution=(1, 1),
            array_out=None,
            array_out_win=None,
            nodata_out=nodata_out,
            array_in_origin=(0.0, 0.0),
            win=None,
            array_in_mask=mask,
            grid_mask=grid_mask,
            grid_mask_valid_value=grid_mask_valid_value,
            grid_nodata=None,
            array_out_mask=mask is not None,
            check_boundaries=check_boundaries,
            standalone=standalone,
            boundary_condition=boundary_condition,
        )
        if mask is None:
            # test resampled data
            np.testing.assert_array_almost_equal(
                array,
                array_out,
                decimal=testing_decimal,
            )
            # test mask
            assert mask_out is None

        else:
            # A mask was provided
            # The output mask should be set and identical to the input mask
            np.testing.assert_array_equal(mask, mask_out)

            # The output image should be set to nodata_out for masked coordinates
            masked_indices = np.where(mask != UNMASKED_VALUE)
            unmasked_indices = np.where(mask == UNMASKED_VALUE)

            if array.ndim == 2:
                assert np.all(array_out[masked_indices] == nodata_out)
                np.testing.assert_array_almost_equal(
                    array[unmasked_indices],
                    array_out[unmasked_indices],
                    decimal=testing_decimal,
                )
            else:
                for i in range(array.shape[0]):
                    assert np.all(array_out[i][masked_indices] == nodata_out)
                    np.testing.assert_array_almost_equal(
                        array[i][unmasked_indices],
                        array_out[i][unmasked_indices],
                        decimal=testing_decimal,
                    )

    @pytest.mark.parametrize(
        "array, grid, interp, interp_kwargs, mask, grid_mask, grid_mask_valid_value, grid_nodata, "
        "check_boundaries, testing_decimal",
        [
            (
                ARRAY_IN_001_ARRAY,
                GRID_IN_001_01_GRID,
                "nearest",
                None,
                None,
                None,
                None,
                None,
                True,
                6,
            ),
            (
                ARRAY_IN_001_ARRAY,
                GRID_IN_001_01_GRID,
                "nearest",
                None,
                MASK_IN_001_ARRAY_01,
                None,
                None,
                None,
                True,
                6,
            ),
            (
                ARRAY_IN_001_ARRAY,
                GRID_IN_001_01_GRID,
                "nearest",
                None,
                MASK_IN_001_ARRAY_02,
                None,
                None,
                None,
                True,
                6,
            ),
            (
                ARRAY_IN_001_ARRAY,
                GRID_IN_001_01_GRID,
                "nearest",
                None,
                None,
                None,
                None,
                None,
                False,
                6,
            ),  # with nearest idendity we can deactivate boundaries check
            (
                ARRAY_IN_001_ARRAY,
                GRID_IN_001_01_GRID,
                "linear",
                None,
                None,
                None,
                None,
                None,
                True,
                6,
            ),
            (
                ARRAY_IN_001_ARRAY,
                GRID_IN_001_01_GRID,
                "linear",
                None,
                MASK_IN_001_ARRAY_01,
                None,
                None,
                None,
                True,
                6,
            ),
            (
                ARRAY_IN_001_ARRAY,
                GRID_IN_001_01_GRID,
                "linear",
                None,
                MASK_IN_001_ARRAY_02,
                None,
                None,
                None,
                True,
                6,
            ),
            (
                ARRAY_IN_001_ARRAY,
                GRID_IN_001_01_GRID,
                "cubic",
                None,
                None,
                None,
                None,
                None,
                True,
                6,
            ),
            (
                ARRAY_IN_001_ARRAY,
                GRID_IN_001_01_GRID,
                "cubic",
                None,
                MASK_IN_001_ARRAY_01,
                None,
                None,
                None,
                True,
                6,
            ),
            (
                ARRAY_IN_001_ARRAY,
                GRID_IN_001_01_GRID,
                "cubic",
                None,
                MASK_IN_001_ARRAY_02,
                None,
                None,
                None,
                True,
                6,
            ),
            (
                np.asarray([ARRAY_IN_001_ARRAY, 2 * ARRAY_IN_001_ARRAY]),
                GRID_IN_001_01_GRID,
                "cubic",
                None,
                MASK_IN_001_ARRAY_02,
                None,
                None,
                None,
                True,
                6,
            ),
            (
                ARRAY_IN_001_ARRAY,
                GRID_IN_001_01_GRID,
                "bspline3",
                {"epsilon": 1e-6, "mask_influence_threshold": 1},
                MASK_IN_001_ARRAY_02,
                None,
                None,
                None,
                True,
                6,
            ),
            (
                np.asarray([ARRAY_IN_001_ARRAY, 2 * ARRAY_IN_001_ARRAY]),
                GRID_IN_001_01_GRID,
                "bspline3",
                {"epsilon": 1e-6, "mask_influence_threshold": 1},
                MASK_IN_001_ARRAY_02,
                None,
                None,
                None,
                True,
                6,
            ),
            (
                ARRAY_IN_001_ARRAY,
                GRID_IN_001_01_GRID,
                get_interpolator("bspline5", epsilon=1e-6, mask_influence_threshold=1),
                None,
                MASK_IN_001_ARRAY_02,
                None,
                None,
                None,
                True,
                6,
            ),
        ],
    )
    def test_grid_resampling_standalone_yes_idendity_1_1(
        self,
        array,
        grid,
        interp,
        interp_kwargs,
        mask,
        grid_mask,
        grid_mask_valid_value,
        grid_nodata,
        check_boundaries,
        testing_decimal,
    ):
        """Test idendity resampling with a full resolute grid"""
        nodata_out = -10
        standalone = True
        boundary_condition = "symmetric"

        array_out, mask_out = array_grid_resampling(
            interp=interp,
            interp_kwargs=interp_kwargs,
            array_in=array,
            grid_row=grid[0],
            grid_col=grid[1],
            grid_resolution=(1, 1),
            array_out=None,
            array_out_win=None,
            nodata_out=nodata_out,
            array_in_origin=(0.0, 0.0),
            win=None,
            array_in_mask=mask,
            grid_mask=grid_mask,
            grid_mask_valid_value=grid_mask_valid_value,
            grid_nodata=None,
            array_out_mask=mask is not None,
            check_boundaries=check_boundaries,
            standalone=standalone,
            boundary_condition=boundary_condition,
        )
        if mask is None:
            # test resampled data
            np.testing.assert_array_almost_equal(
                array,
                array_out,
                decimal=testing_decimal,
            )
            # test mask
            assert mask_out is None

        else:
            # A mask was provided
            # The output mask should be set and identical to the input mask
            np.testing.assert_array_equal(mask, mask_out)

            # The output image should be set to nodata_out for masked coordinates
            masked_indices = np.where(mask != UNMASKED_VALUE)
            unmasked_indices = np.where(mask == UNMASKED_VALUE)

            if array.ndim == 2:
                assert np.all(array_out[masked_indices] == nodata_out)
                np.testing.assert_array_almost_equal(
                    array[unmasked_indices],
                    array_out[unmasked_indices],
                    decimal=testing_decimal,
                )
            else:
                for i in range(array.shape[0]):
                    assert np.all(array_out[i][masked_indices] == nodata_out)
                    np.testing.assert_array_almost_equal(
                        array[i][unmasked_indices],
                        array_out[i][unmasked_indices],
                        decimal=testing_decimal,
                    )

    @pytest.mark.parametrize(
        "interp, interp_kwargs",
        [
            ("nearest", None),
            ("linear", None),
            ("cubic", None),
            ("bspline3", {"epsilon": 1e-6, "mask_influence_threshold": 1}),
        ],
    )
    @pytest.mark.parametrize("boundary_condition", ["reflect", "symmetric", "wrap"])
    def test_grid_resampling_standalone_yes_boundary_condition(
        self,
        interp,
        interp_kwargs,
        boundary_condition,
    ):
        """Check standalone mode for non None boundary condition"""
        if isinstance(interp, str):
            if interp_kwargs is None:
                interp_kwargs = {}
            interp = get_interpolator(interp, **interp_kwargs)
            interp.initialize()

        margins = interp.total_margins()
        nodata_out = 666

        for do_mask in (True, False):
            array_in = np.arange((5 * 7), dtype=np.float64).reshape((5, 7))
            grid = create_grid(
                nrow=2,
                ncol=2,
                origin_pos=(0, 0),
                origin_node=(0, 0.1),
                v_row_y=1,
                v_row_x=0,
                v_col_y=0,
                v_col_x=1,
                grid_dtype=np.float64,
            )
            mask = None
            if do_mask:
                mask = np.ones(array_in.shape, dtype=np.uint8)
                mask[0, 0] = 0

            array_in_bis = np.pad(
                array_in, np.asarray(margins).reshape((2, 2)), mode=boundary_condition
            )
            grid_bis = np.copy(grid)
            grid_bis[0] += margins[0]
            grid_bis[1] += margins[2]
            mask_bis = None
            if do_mask:
                mask_bis = np.pad(
                    mask, np.asarray(margins).reshape((2, 2)), mode=boundary_condition
                )

            # First resampling with standalone and boundary_condition
            array_out, mask_out = array_grid_resampling(
                interp=interp,
                array_in=array_in,
                grid_row=grid[0],
                grid_col=grid[1],
                grid_resolution=(1, 1),
                array_out=None,
                array_out_win=None,
                nodata_out=nodata_out,
                array_in_origin=(0.0, 0.0),
                win=None,
                array_in_mask=mask,
                grid_mask=None,
                grid_mask_valid_value=None,
                grid_nodata=None,
                array_out_mask=mask is not None,
                check_boundaries=True,
                standalone=True,
                boundary_condition=boundary_condition,
            )

            array_out_bis, mask_out_bis = array_grid_resampling(
                interp=interp,
                interp_kwargs=interp_kwargs,
                array_in=array_in_bis,
                grid_row=grid_bis[0],
                grid_col=grid_bis[1],
                grid_resolution=(1, 1),
                array_out=None,
                array_out_win=None,
                nodata_out=nodata_out,
                array_in_origin=(0.0, 0.0),
                win=None,
                array_in_mask=mask_bis,
                grid_mask=None,
                grid_mask_valid_value=None,
                grid_nodata=None,
                array_out_mask=mask_bis is not None,
                check_boundaries=True,
                standalone=True,
                boundary_condition=None,
            )

            np.testing.assert_array_almost_equal(array_out, array_out_bis, 1e-6)
            if mask is not None:
                np.testing.assert_array_almost_equal(mask_out, mask_out_bis, 1e-6)
