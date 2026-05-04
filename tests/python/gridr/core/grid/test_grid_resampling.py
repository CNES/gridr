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
from typing import Optional, Tuple

import numpy as np
import pytest

from gridr.core.grid.grid_mask import Validity
from gridr.core.grid.grid_resampling import (  # standalone_preprocessing, TODO
    ResamplingMaskStrategy,
    apply_mask_strategy,
    array_grid_resampling,
    check_mask_strategy,
    get_array_padded_shape,
    resolve_mask_strategy,
    source_extent_pad,
)
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


def optional_array_equal(data1, data2):
    test_equal = False
    if data1 is None and data2 is None:
        test_equal = True
    elif data1 is None or data2 is None:
        test_equal = False
    else:
        test_equal = np.array_equal(data1, data2)
    return test_equal


class TestGridResampling:
    """Test class"""

    # ---------------------------------------------------------------------------
    # Test get_array_padded_shape
    # ---------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "array_src, pad, expected_shape, expected_window",
        [
            # 2D tests
            (
                np.ones((100, 100)),
                ((5, 5), (10, 10)),
                (110, 120),
                (slice(5, 105), slice(10, 110)),
            ),
            (
                np.ones((50, 60)),
                ((0, 0), (0, 0)),
                (50, 60),
                (slice(0, 50), slice(0, 60)),
            ),
            (
                np.ones((10, 10)),
                ((2, 8), (3, 7)),
                (20, 20),
                (slice(2, 12), slice(3, 13)),
            ),
            (
                np.ones((1, 1)),
                ((100, 100), (100, 100)),
                (201, 201),
                (slice(100, 101), slice(100, 101)),
            ),
            (
                np.ones((1, 1)),
                ((0, 0), (0, 0)),
                (1, 1),
                (slice(0, 1), slice(0, 1)),
            ),
            # 3D tests
            (
                np.ones((3, 100, 100)),
                ((5, 5), (10, 10)),
                (3, 110, 120),
                (slice(None), slice(5, 105), slice(10, 110)),
            ),
            (
                np.ones((5, 20, 30)),
                ((1, 2), (3, 4)),
                (5, 23, 37),
                (slice(None), slice(1, 21), slice(3, 33)),
            ),
            (
                np.ones((2, 10, 20)),
                ((0, 0), (0, 0)),
                (2, 10, 20),
                (slice(None), slice(0, 10), slice(0, 20)),
            ),
            (
                np.ones((1, 1, 1)),
                ((0, 0), (0, 0)),
                (1, 1, 1),
                (slice(None), slice(0, 1), slice(0, 1)),
            ),
            (
                np.ones((10, 50, 50)),
                ((5, 5), (10, 10)),
                (10, 60, 70),
                (slice(None), slice(5, 55), slice(10, 60)),
            ),
        ],
    )
    def test_get_array_padded_shape(self, array_src, pad, expected_shape, expected_window):
        """Test get_array_padded_shape with various inputs."""
        shape, window = get_array_padded_shape(array_src, pad)
        assert window == expected_window
        assert shape == expected_shape

    @pytest.mark.parametrize(
        "array_src, pad",
        [
            (np.ones((10, 10)), ((5, 6), (10, 11))),
            (np.ones((1, 10, 10)), ((5, 6), (10, 11))),
            (np.ones((3, 10, 10)), ((5, 6), (10, 11))),
        ],
    )
    def test_get_array_padded_shape_matches_pad(self, array_src, pad):
        """Verify computed shape matches actual np.pad result."""
        shape, _ = get_array_padded_shape(array_src, pad)

        if array_src.ndim == 2:
            padded = np.pad(array_src, pad_width=pad, mode="constant")
        else:
            # 3D: pad only last 2 dims
            padded = np.pad(array_src, pad_width=((0, 0), pad[0], pad[1]), mode="constant")

        assert shape == padded.shape

    @pytest.mark.parametrize(
        "array_src, pad",
        [
            (np.ones((50,)), ((5, 5), (10, 10))),  # 1D
            (np.ones((2, 3, 10, 10)), ((5, 5), (10, 10))),  # 4D
            (np.array(5), ((5, 5), (10, 10))),  # Scalar
        ],
    )
    def test_get_array_padded_shape_invalid_dims(self, array_src, pad):
        """Test that invalid dimension arrays raise ValueError."""
        with pytest.raises(ValueError, match="2 or 3 dimensions"):
            get_array_padded_shape(array_src, pad)

    # ---------------------------------------------------------------------------
    # Test source_extent_pad
    # ---------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "array_src ,pad, boundary_condition, fill",
        [
            # 2D - edge boundary
            (np.ones((10, 10)), ((1, 2), (3, 4)), "edge", None),
            (np.arange(100).reshape(10, 10), ((1, 2), (3, 4)), "edge", None),
            (np.arange(100).reshape(10, 10), ((0, 0), (0, 0)), "edge", None),
            (np.arange(100).reshape(10, 10), ((2, 2), (2, 2)), "edge", 999),
            # 2D - reflect boundary
            (np.ones((10, 10)), ((1, 1), (1, 1)), "reflect", None),
            # 2D - symmetric boundary
            (np.ones((10, 10)), ((1, 1), (1, 1)), "symmetric", None),
            # 2D - wrap boundary
            (np.ones((10, 10)), ((1, 1), (1, 1)), "wrap", None),
            # 2D - fill only
            (np.ones((10, 10)), ((2, 2), (2, 2)), None, -999),
            # 2D - both fill and boundary
            (np.ones((10, 10)), ((1, 1), (1, 1)), "edge", 0.0),
            # 3D - edge boundary
            (np.ones((3, 10, 10)), ((5, 5), (5, 5)), "edge", None),
            (np.ones((5, 20, 30)), ((2, 2), (3, 3)), "edge", None),
            # 3D - fill only
            (np.ones((3, 10, 10)), ((2, 2), (2, 2)), None, -999),
        ],
    )
    def test_source_extent_pad_vs_np_pad(self, array_src, pad, boundary_condition, fill):
        """Compare source_extent_pad with np.pad for 2D and 3D arrays."""
        result = source_extent_pad(array_src, pad, boundary_condition, fill)

        np_kwargs = {}
        if boundary_condition is None:
            boundary_condition = "constant"
            np_kwargs["constant_values"] = fill if fill is not None else 0

        # Check shape matches np.pad
        if array_src.ndim == 2:
            np_padded = np.pad(array_src, pad_width=pad, mode=boundary_condition, **np_kwargs)
        else:
            # For 3D: pad only last 2 dims
            np_pad_width = ((0, 0),) + pad
            np_mode = boundary_condition
            np_padded = np.pad(array_src, pad_width=np_pad_width, mode=np_mode, **np_kwargs)

        assert result.shape == np_padded.shape
        assert result.flags["C_CONTIGUOUS"]
        assert result.dtype == array_src.dtype
        assert np.array_equal(result, np_padded)

    # ---------------------------------------------------------------------------
    # Test resolve_mask_strategy
    # ---------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "interp, pad, array_in_mask, boundary_condition, trust_padding, "
        "array_in_shape, array_in_mask_safe_win, expected",
        [
            # no pad, no mask, no boundary_condition, no trust
            # 0
            (
                get_interpolator("cubic"),
                np.array([[0, 0], [0, 0]]),
                None,
                None,
                False,
                (3, 2, 3),
                None,
                ResamplingMaskStrategy(
                    mask_kind="none",
                    safe_region=None,
                    needs_mask_alloc=False,
                    pad_fill=None,
                    boundary_condition=None,
                ),
            ),
            # 1
            (
                get_interpolator("cubic"),
                np.array([[0, 0], [0, 0]]),
                np.ones((2, 3), dtype=np.uint8),
                None,
                False,
                (3, 2, 3),
                None,
                ResamplingMaskStrategy(
                    mask_kind="none",
                    safe_region=None,
                    needs_mask_alloc=False,
                    pad_fill=None,
                    boundary_condition=None,
                ),
            ),
            # no pad, no mask, no boundary_condition, no trust - bspline
            # 2
            (
                get_interpolator("bspline3", epsilon=1e-6, mask_influence_threshold=1),
                np.array([[0, 0], [0, 0]]),
                None,
                None,
                False,
                (3, 2, 3),
                None,
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=np.array([[0, 1], [0, 2]]),
                    needs_mask_alloc=True,
                    pad_fill=Validity.INVALID,
                    boundary_condition=None,
                ),
            ),
            # 3
            (
                get_interpolator("bspline3", epsilon=1e-6, mask_influence_threshold=1),
                np.array([[0, 0], [0, 0]]),
                np.ones((2, 3), dtype=np.uint8),
                None,
                False,
                (3, 2, 3),
                None,
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=np.array([[0, 1], [0, 2]]),
                    needs_mask_alloc=False,
                    pad_fill=None,
                    boundary_condition=None,
                ),
            ),
            # 4
            # no pad, mask, no boundary_condition, no trust, no safe win
            (
                get_interpolator("cubic"),
                np.array([[0, 0], [0, 0]]),
                np.array(
                    [
                        [0, 1, 1],
                        [1, 1, 1],
                    ]
                ),
                None,
                False,
                (3, 2, 3),
                None,
                ResamplingMaskStrategy(
                    mask_kind="binary",
                    safe_region=None,
                    needs_mask_alloc=False,
                    pad_fill=None,
                    boundary_condition=None,
                ),
            ),
            # no pad, mask, no boundary_condition, no trust, no safe win
            # bspline
            # 5
            (
                get_interpolator("bspline3", epsilon=1e-6, mask_influence_threshold=1),
                np.array([[0, 0], [0, 0]]),
                np.array(
                    [
                        [0, 1, 1],
                        [1, 1, 1],
                    ]
                ),
                None,
                False,
                (3, 2, 3),
                None,
                ResamplingMaskStrategy(
                    mask_kind="binary",
                    safe_region=None,
                    needs_mask_alloc=False,
                    pad_fill=None,
                    boundary_condition=None,
                ),
            ),
            # no pad, mask, no boundary_condition, no trust, safe win
            # 6
            (
                get_interpolator("cubic"),
                np.array([[0, 0], [0, 0]]),
                np.array(
                    [
                        [0, 1, 1],
                        [1, 1, 1],
                    ]
                ),
                None,
                False,
                (3, 10, 11),
                ((1, 1), (1, 2)),
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=((1, 1), (1, 2)),
                    needs_mask_alloc=False,
                    pad_fill=None,
                    boundary_condition=None,
                ),
            ),
            # no pad, mask, no boundary_condition, no trust, safe win - bspline
            # 7
            (
                get_interpolator("bspline3", epsilon=1e-6, mask_influence_threshold=1),
                np.array([[0, 0], [0, 0]]),
                np.array(
                    [
                        [0, 1, 1],
                        [1, 1, 1],
                    ]
                ),
                None,
                False,
                (3, 10, 11),
                ((1, 1), (1, 2)),
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=((1, 1), (1, 2)),
                    needs_mask_alloc=False,
                    pad_fill=None,
                    boundary_condition=None,
                ),
            ),
            # no pad, mask, boundary_condition, trust, safe win
            # 8
            (
                get_interpolator("cubic"),
                np.array([[0, 0], [0, 0]]),
                np.array(
                    [
                        [0, 1, 1],
                        [1, 1, 1],
                    ]
                ),
                "reflect",
                True,
                (3, 2, 3),
                ((1, 1), (1, 2)),
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=((1, 1), (1, 2)),
                    needs_mask_alloc=False,
                    pad_fill=None,
                    boundary_condition=None,
                ),
            ),
            # no pad, mask, boundary_condition, trust, safe win - bspline
            # 9
            (
                get_interpolator("bspline3", epsilon=1e-6, mask_influence_threshold=1),
                np.array([[0, 0], [0, 0]]),
                np.array(
                    [
                        [0, 1, 1],
                        [1, 1, 1],
                    ]
                ),
                "reflect",
                True,
                (3, 2, 3),
                ((1, 1), (1, 2)),
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=((1, 1), (1, 2)),
                    needs_mask_alloc=False,
                    pad_fill=None,
                    boundary_condition=None,
                ),
            ),
            # pad, mask, boundary_condition, trust, safe win
            # 10
            (
                get_interpolator("cubic"),
                np.array([[2, 3], [1, 4]]),
                np.array(
                    [
                        [0, 1, 1],
                        [1, 1, 1],
                    ]
                ),
                "reflect",
                True,
                (3, 2, 3),
                ((1, 1), (1, 2)),
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=((3, 3), (2, 3)),
                    needs_mask_alloc=False,
                    pad_fill=Validity.VALID,
                    boundary_condition="reflect",
                ),
            ),
            # pad, mask, boundary_condition, trust, safe  - bspline
            # 11
            (
                get_interpolator("bspline3", epsilon=1e-6, mask_influence_threshold=1),
                np.array([[2, 3], [1, 4]]),
                np.array(
                    [
                        [0, 1, 1],
                        [1, 1, 1],
                    ]
                ),
                "reflect",
                True,
                (3, 2, 3),
                ((1, 1), (1, 2)),
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=((3, 3), (2, 3)),
                    needs_mask_alloc=False,
                    pad_fill=Validity.VALID,
                    boundary_condition="reflect",
                ),
            ),
            # pad, mask, boundary_condition, no trust, safe win
            # 12
            (
                get_interpolator("cubic"),
                np.array([[2, 3], [1, 4]]),
                np.array(
                    [
                        [0, 1, 1],
                        [1, 1, 1],
                    ]
                ),
                "reflect",
                False,
                (3, 2, 3),
                ((1, 1), (1, 2)),
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=((3, 3), (2, 3)),
                    needs_mask_alloc=False,
                    pad_fill=Validity.INVALID,
                    boundary_condition=None,
                ),
            ),
            # pad, mask, boundary_condition, no trust, safe win - bspline
            # 13
            (
                get_interpolator("bspline3", epsilon=1e-6, mask_influence_threshold=1),
                np.array([[2, 3], [1, 4]]),
                np.array(
                    [
                        [0, 1, 1],
                        [1, 1, 1],
                    ]
                ),
                "reflect",
                False,
                (3, 2, 3),
                ((1, 1), (1, 2)),
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=((3, 3), (2, 3)),
                    needs_mask_alloc=False,
                    pad_fill=Validity.INVALID,
                    boundary_condition=None,
                ),
            ),
            # pad, mask, boundary_condition, no trust, safe win
            # 14
            (
                get_interpolator("cubic"),
                np.array([[2, 3], [1, 4]]),
                None,
                "reflect",
                False,
                (3, 2, 3),
                ((8, 8), (100, 100)),  # must be ignored
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=((2, 3), (1, 3)),
                    needs_mask_alloc=True,
                    pad_fill=Validity.INVALID,
                    boundary_condition=None,
                ),
            ),
            # 15
            (
                get_interpolator("cubic"),
                np.array([[2, 3], [1, 4]]),
                np.array(
                    [
                        [1, 1, 1],
                        [1, 1, 1],
                    ]
                ),
                "reflect",
                False,
                (3, 2, 3),
                ((8, 8), (100, 100)),  # must be ignored
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=((2, 3), (1, 3)),
                    needs_mask_alloc=False,
                    pad_fill=Validity.INVALID,
                    boundary_condition=None,
                ),
            ),
            # pad, mask, boundary_condition, no trust, safe win - bspline
            # 16
            (
                get_interpolator("bspline3", epsilon=1e-6, mask_influence_threshold=1),
                np.array([[2, 3], [1, 4]]),
                None,
                "reflect",
                False,
                (3, 2, 3),
                ((8, 8), (100, 100)),  # must be ignored
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=((2, 3), (1, 3)),
                    needs_mask_alloc=True,
                    pad_fill=Validity.INVALID,
                    boundary_condition=None,
                ),
            ),
            # 17
            (
                get_interpolator("bspline3", epsilon=1e-6, mask_influence_threshold=1),
                np.array([[2, 3], [1, 4]]),
                np.array(
                    [
                        [1, 1, 1],
                        [1, 1, 1],
                    ]
                ),
                "reflect",
                False,
                (3, 2, 3),
                ((8, 8), (100, 100)),  # must be ignored
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=((2, 3), (1, 3)),
                    needs_mask_alloc=False,
                    pad_fill=Validity.INVALID,
                    boundary_condition=None,
                ),
            ),
            # pad, no mask, boundary_condition, trust, safe win
            # 18
            (
                get_interpolator("cubic"),
                np.array([[2, 3], [1, 4]]),
                None,
                "reflect",
                True,
                (3, 2, 3),
                ((8, 8), (100, 100)),  # must be ignored
                ResamplingMaskStrategy(
                    mask_kind="none",
                    safe_region=None,
                    needs_mask_alloc=False,
                    pad_fill=None,
                    boundary_condition=None,
                ),
            ),
            # 19
            (
                get_interpolator("cubic"),
                np.array([[2, 3], [1, 4]]),
                np.array(
                    [
                        [1, 1, 1],
                        [1, 1, 1],
                    ]
                ),
                "reflect",
                True,
                (3, 2, 3),
                ((8, 8), (100, 100)),  # must be ignored
                ResamplingMaskStrategy(
                    mask_kind="none",
                    safe_region=None,
                    needs_mask_alloc=False,
                    pad_fill=None,
                    boundary_condition=None,
                ),
            ),
            # pad, mask, boundary_condition, trust, safe win
            # bspline interpolation must force mask creation because of
            # data invalidation during prefiltering
            # 20
            (
                get_interpolator("bspline3", epsilon=1e-6, mask_influence_threshold=1),
                np.array([[2, 3], [1, 4]]),
                None,
                "reflect",
                True,
                (3, 2, 3),
                ((8, 8), (100, 100)),  # must be ignored
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=(
                        (0, 6),
                        (0, 7),
                    ),  # no input mask / trust pad / safe_region is all data before prefiltering
                    needs_mask_alloc=True,
                    pad_fill=Validity.VALID,
                    boundary_condition=None,
                ),
            ),
            # 21
            (
                get_interpolator("bspline3", epsilon=1e-6, mask_influence_threshold=1),
                np.array([[2, 3], [1, 4]]),
                np.array(
                    [
                        [1, 1, 1],
                        [1, 1, 1],
                    ]
                ),
                "reflect",
                True,
                (3, 2, 3),
                ((8, 8), (100, 100)),  # must be ignored
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=(
                        (0, 6),
                        (0, 7),
                    ),  # no input mask / trust pad / safe_region is all data before prefiltering
                    needs_mask_alloc=False,
                    pad_fill=Validity.VALID,
                    boundary_condition="reflect",
                ),
            ),
            # pad, no mask, boundary_condition, no trust, safe win
            # 22
            (
                get_interpolator("cubic"),
                np.array([[2, 3], [1, 4]]),
                None,
                "reflect",
                False,
                (3, 2, 3),
                ((8, 8), (100, 100)),  # must be ignored
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=((2, 3), (1, 3)),
                    needs_mask_alloc=True,
                    pad_fill=Validity.INVALID,
                    boundary_condition=None,
                ),
            ),
            # 23
            (
                get_interpolator("cubic"),
                np.array([[2, 3], [1, 4]]),
                np.array(
                    [
                        [1, 1, 1],
                        [1, 1, 1],
                    ]
                ),
                "reflect",
                False,
                (3, 2, 3),
                ((8, 8), (100, 100)),  # must be ignored
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=((2, 3), (1, 3)),
                    needs_mask_alloc=False,
                    pad_fill=Validity.INVALID,
                    boundary_condition=None,
                ),
            ),
            # pad, no mask, boundary_condition, no trust, safe win
            # bspline interpolation must force mask creation because of
            # data invalidation during prefiltering
            # 24
            (
                get_interpolator("bspline3", epsilon=1e-6, mask_influence_threshold=1),
                np.array([[2, 3], [1, 4]]),
                None,
                "reflect",
                False,
                (3, 2, 3),
                ((8, 8), (100, 100)),  # must be ignored
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=((2, 3), (1, 3)),
                    needs_mask_alloc=True,
                    pad_fill=Validity.INVALID,
                    boundary_condition=None,
                ),
            ),
            # 25
            (
                get_interpolator("bspline3", epsilon=1e-6, mask_influence_threshold=1),
                np.array([[2, 3], [1, 4]]),
                np.array(
                    [
                        [1, 1, 1],
                        [1, 1, 1],
                    ]
                ),
                "reflect",
                False,
                (3, 2, 3),
                ((8, 8), (100, 100)),  # must be ignored
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=((2, 3), (1, 3)),
                    needs_mask_alloc=False,
                    pad_fill=Validity.INVALID,
                    boundary_condition=None,
                ),
            ),
            # pad, no mask, no boundary_condition, no trust, safe win
            # 26
            (
                get_interpolator("cubic"),
                np.array([[2, 3], [1, 4]]),
                None,
                None,
                False,
                (3, 2, 3),
                ((8, 8), (100, 100)),  # must be ignored
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=((2, 3), (1, 3)),
                    needs_mask_alloc=True,
                    pad_fill=Validity.INVALID,
                    boundary_condition=None,
                ),
            ),
            # 27
            (
                get_interpolator("cubic"),
                np.array([[2, 3], [1, 4]]),
                np.array(
                    [
                        [1, 1, 1],
                        [1, 1, 1],
                    ]
                ),
                None,
                False,
                (3, 2, 3),
                ((8, 8), (100, 100)),  # must be ignored
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=((2, 3), (1, 3)),
                    needs_mask_alloc=False,
                    pad_fill=Validity.INVALID,
                    boundary_condition=None,
                ),
            ),
            # pad, no mask, no boundary_condition, no trust, safe win
            # bspline interpolation must force mask creation because of
            # data invalidation during prefiltering
            # 28
            (
                get_interpolator("bspline3", epsilon=1e-6, mask_influence_threshold=1),
                np.array([[2, 3], [1, 4]]),
                None,
                None,
                False,
                (3, 2, 3),
                ((8, 8), (100, 100)),  # must be ignored
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=((2, 3), (1, 3)),
                    needs_mask_alloc=True,
                    pad_fill=Validity.INVALID,
                    boundary_condition=None,
                ),
            ),
            # 29
            (
                get_interpolator("bspline3", epsilon=1e-6, mask_influence_threshold=1),
                np.array([[2, 3], [1, 4]]),
                np.array(
                    [
                        [1, 1, 1],
                        [1, 1, 1],
                    ]
                ),
                None,
                False,
                (3, 2, 3),
                ((8, 8), (100, 100)),  # must be ignored
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=((2, 3), (1, 3)),
                    needs_mask_alloc=False,
                    pad_fill=Validity.INVALID,
                    boundary_condition=None,
                ),
            ),
        ],
    )
    def test_resolve_mask_strategy(
        self,
        interp,
        pad,
        array_in_mask,
        boundary_condition,
        trust_padding,
        array_in_shape,
        array_in_mask_safe_win,
        expected,
    ):
        """Test resolve_mask_strategy and check it"""
        strategy = resolve_mask_strategy(
            interp,
            pad,
            array_in_mask,
            boundary_condition,
            trust_padding,
            array_in_shape,
            array_in_mask_safe_win,
        )
        # Check the resolved strategy is OK
        check_mask_strategy(pad, strategy)
        assert strategy.mask_kind == expected.mask_kind
        assert strategy.needs_mask_alloc == expected.needs_mask_alloc
        assert strategy.pad_fill == expected.pad_fill
        assert strategy.boundary_condition == expected.boundary_condition
        assert optional_array_equal(strategy.safe_region, expected.safe_region)

    @pytest.mark.parametrize(
        "pad, strategy, expected",
        [
            # ------------------------------------------------------------------
            # PAD = NO - MASK_KIND = "none"
            # ------------------------------------------------------------------
            # pad = no | mask_kind = "none" | safe_region = "none"
            # 0
            (
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="none",
                    safe_region=None,
                    needs_mask_alloc=False,
                    pad_fill=None,
                    boundary_condition=None,
                ),
                None,
            ),
            # 1
            (
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="none",
                    safe_region=None,
                    needs_mask_alloc=False,
                    pad_fill=None,
                    boundary_condition="reflect",
                ),
                ValueError,
            ),
            # 2
            (
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="none",
                    safe_region=None,
                    needs_mask_alloc=False,
                    pad_fill=Validity.VALID,
                    boundary_condition=None,
                ),
                ValueError,
            ),
            # 3
            (
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="none",
                    safe_region=None,
                    needs_mask_alloc=False,
                    pad_fill=Validity.VALID,
                    boundary_condition="reflect",
                ),
                ValueError,
            ),
            # 4
            (
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="none",
                    safe_region=None,
                    needs_mask_alloc=True,
                    pad_fill=None,
                    boundary_condition=None,
                ),
                ValueError,
            ),
            # 5
            (
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="none",
                    safe_region=None,
                    needs_mask_alloc=True,
                    pad_fill=None,
                    boundary_condition="reflect",
                ),
                ValueError,
            ),
            # 6
            (
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="none",
                    safe_region=None,
                    needs_mask_alloc=True,
                    pad_fill=Validity.VALID,
                    boundary_condition=None,
                ),
                ValueError,
            ),
            # 7
            (
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="none",
                    safe_region=None,
                    needs_mask_alloc=True,
                    pad_fill=Validity.VALID,
                    boundary_condition="reflect",
                ),
                ValueError,
            ),
            # pad = no | mask_kind = "none" | safe_region != "none"
            # 8
            (
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="none",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=False,
                    pad_fill=None,
                    boundary_condition=None,
                ),
                ValueError,
            ),
            # 9
            (
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="none",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=False,
                    pad_fill=None,
                    boundary_condition="reflect",
                ),
                ValueError,
            ),
            # 10
            (
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="none",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=False,
                    pad_fill=Validity.VALID,
                    boundary_condition=None,
                ),
                ValueError,
            ),
            # 11
            (
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="none",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=False,
                    pad_fill=Validity.VALID,
                    boundary_condition="reflect",
                ),
                ValueError,
            ),
            # 12
            (
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="none",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=True,
                    pad_fill=None,
                    boundary_condition=None,
                ),
                ValueError,
            ),
            # 13
            (
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="none",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=True,
                    pad_fill=None,
                    boundary_condition="reflect",
                ),
                ValueError,
            ),
            # 14
            (
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="none",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=True,
                    pad_fill=Validity.VALID,
                    boundary_condition=None,
                ),
                ValueError,
            ),
            # 15
            (
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="none",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=True,
                    pad_fill=Validity.VALID,
                    boundary_condition="reflect",
                ),
                ValueError,
            ),
            # ------------------------------------------------------------------
            # PAD = NO - MASK_KIND = "safe_region"
            # ------------------------------------------------------------------
            # pad = no | mask_kind = "safe_region" | safe_region = "none"
            # 16
            (
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=None,
                    needs_mask_alloc=False,
                    pad_fill=None,
                    boundary_condition=None,
                ),
                ValueError,
            ),
            # 17
            (
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=None,
                    needs_mask_alloc=False,
                    pad_fill=None,
                    boundary_condition="reflect",
                ),
                ValueError,
            ),
            # 18
            (
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=None,
                    needs_mask_alloc=False,
                    pad_fill=Validity.VALID,
                    boundary_condition=None,
                ),
                ValueError,
            ),
            # 19
            (
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=None,
                    needs_mask_alloc=False,
                    pad_fill=Validity.VALID,
                    boundary_condition="reflect",
                ),
                ValueError,
            ),
            # 20
            (
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=None,
                    needs_mask_alloc=True,
                    pad_fill=None,
                    boundary_condition=None,
                ),
                ValueError,
            ),
            # 21
            (
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=None,
                    needs_mask_alloc=True,
                    pad_fill=None,
                    boundary_condition="reflect",
                ),
                ValueError,
            ),
            # 22
            (
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=None,
                    needs_mask_alloc=True,
                    pad_fill=Validity.VALID,
                    boundary_condition=None,
                ),
                ValueError,
            ),
            # 23
            (
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=None,
                    needs_mask_alloc=True,
                    pad_fill=Validity.VALID,
                    boundary_condition="reflect",
                ),
                ValueError,
            ),
            # pad = no | mask_kind = "safe_region" | safe_region != "none"
            # 24
            (
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=False,
                    pad_fill=None,
                    boundary_condition=None,
                ),
                None,
            ),
            # 25
            (
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=False,
                    pad_fill=None,
                    boundary_condition="reflect",
                ),
                ValueError,
            ),
            # 26
            (
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=False,
                    pad_fill=Validity.VALID,
                    boundary_condition=None,
                ),
                ValueError,
            ),
            # 27
            (
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=False,
                    pad_fill=Validity.VALID,
                    boundary_condition="reflect",
                ),
                ValueError,
            ),
            # 28
            (
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=True,
                    pad_fill=None,
                    boundary_condition=None,
                ),
                ValueError,
            ),
            # 29
            (
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=True,
                    pad_fill=None,
                    boundary_condition="reflect",
                ),
                ValueError,
            ),
            # 30
            (
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=True,
                    pad_fill=Validity.VALID,
                    boundary_condition=None,
                ),
                None,
            ),
            # 31
            (
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=True,
                    pad_fill=Validity.VALID,
                    boundary_condition="reflect",
                ),
                ValueError,
            ),
            # ------------------------------------------------------------------
            # PAD = NO - MASK_KIND = "binary"
            # ------------------------------------------------------------------
            # pad = no | mask_kind = "binary" | safe_region = "none"
            # 32
            (
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="binary",
                    safe_region=None,
                    needs_mask_alloc=False,
                    pad_fill=None,
                    boundary_condition=None,
                ),
                None,
            ),
            # 33
            (
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="binary",
                    safe_region=None,
                    needs_mask_alloc=False,
                    pad_fill=None,
                    boundary_condition="reflect",
                ),
                ValueError,
            ),
            # 34
            (
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="binary",
                    safe_region=None,
                    needs_mask_alloc=False,
                    pad_fill=Validity.VALID,
                    boundary_condition=None,
                ),
                ValueError,
            ),
            # 35
            (
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="binary",
                    safe_region=None,
                    needs_mask_alloc=False,
                    pad_fill=Validity.VALID,
                    boundary_condition="reflect",
                ),
                ValueError,
            ),
            # 36
            (
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="binary",
                    safe_region=None,
                    needs_mask_alloc=True,
                    pad_fill=None,
                    boundary_condition=None,
                ),
                ValueError,
            ),
            # 37
            (
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="binary",
                    safe_region=None,
                    needs_mask_alloc=True,
                    pad_fill=None,
                    boundary_condition="reflect",
                ),
                ValueError,
            ),
            # 38
            (
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="binary",
                    safe_region=None,
                    needs_mask_alloc=True,
                    pad_fill=Validity.VALID,
                    boundary_condition=None,
                ),
                ValueError,
            ),
            # 39
            (
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="binary",
                    safe_region=None,
                    needs_mask_alloc=True,
                    pad_fill=Validity.VALID,
                    boundary_condition="reflect",
                ),
                ValueError,
            ),
            # pad = no | mask_kind = "binary" | safe_region != "none"
            # 40
            (
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="binary",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=False,
                    pad_fill=None,
                    boundary_condition=None,
                ),
                ValueError,
            ),
            # 41
            (
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="binary",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=False,
                    pad_fill=None,
                    boundary_condition="reflect",
                ),
                ValueError,
            ),
            # 42
            (
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="binary",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=False,
                    pad_fill=Validity.VALID,
                    boundary_condition=None,
                ),
                ValueError,
            ),
            # 43
            (
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="binary",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=False,
                    pad_fill=Validity.VALID,
                    boundary_condition="reflect",
                ),
                ValueError,
            ),
            # 44
            (
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="binary",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=True,
                    pad_fill=None,
                    boundary_condition=None,
                ),
                ValueError,
            ),
            # 45
            (
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="binary",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=True,
                    pad_fill=None,
                    boundary_condition="reflect",
                ),
                ValueError,
            ),
            # 46
            (
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="binary",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=True,
                    pad_fill=Validity.VALID,
                    boundary_condition=None,
                ),
                ValueError,
            ),
            # 47
            (
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="binary",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=True,
                    pad_fill=Validity.VALID,
                    boundary_condition="reflect",
                ),
                ValueError,
            ),
            # ------------------------------------------------------------------
            # PAD = YES - MASK_KIND = "none"
            # ------------------------------------------------------------------
            # pad = yes | mask_kind = "none" | safe_region = "none"
            # 48
            (
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="none",
                    safe_region=None,
                    needs_mask_alloc=False,
                    pad_fill=None,
                    boundary_condition=None,
                ),
                None,
            ),
            # 49
            (
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="none",
                    safe_region=None,
                    needs_mask_alloc=False,
                    pad_fill=None,
                    boundary_condition="reflect",
                ),
                ValueError,
            ),
            # 50
            (
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="none",
                    safe_region=None,
                    needs_mask_alloc=False,
                    pad_fill=Validity.VALID,
                    boundary_condition=None,
                ),
                ValueError,
            ),
            # 51
            (
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="none",
                    safe_region=None,
                    needs_mask_alloc=False,
                    pad_fill=Validity.VALID,
                    boundary_condition="reflect",
                ),
                ValueError,
            ),
            # 52
            (
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="none",
                    safe_region=None,
                    needs_mask_alloc=True,
                    pad_fill=None,
                    boundary_condition=None,
                ),
                ValueError,
            ),
            # 53
            (
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="none",
                    safe_region=None,
                    needs_mask_alloc=True,
                    pad_fill=None,
                    boundary_condition="reflect",
                ),
                ValueError,
            ),
            # 54
            (
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="none",
                    safe_region=None,
                    needs_mask_alloc=True,
                    pad_fill=Validity.VALID,
                    boundary_condition=None,
                ),
                ValueError,
            ),
            # 55
            (
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="none",
                    safe_region=None,
                    needs_mask_alloc=True,
                    pad_fill=Validity.VALID,
                    boundary_condition="reflect",
                ),
                ValueError,
            ),
            # pad = yes | mask_kind = "none" | safe_region != "none"
            # 56
            (
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="none",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=False,
                    pad_fill=None,
                    boundary_condition=None,
                ),
                ValueError,
            ),
            # 57
            (
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="none",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=False,
                    pad_fill=None,
                    boundary_condition="reflect",
                ),
                ValueError,
            ),
            # 58
            (
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="none",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=False,
                    pad_fill=Validity.VALID,
                    boundary_condition=None,
                ),
                ValueError,
            ),
            # 59
            (
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="none",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=False,
                    pad_fill=Validity.VALID,
                    boundary_condition="reflect",
                ),
                ValueError,
            ),
            # 60
            (
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="none",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=True,
                    pad_fill=None,
                    boundary_condition=None,
                ),
                ValueError,
            ),
            # 61
            (
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="none",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=True,
                    pad_fill=None,
                    boundary_condition="reflect",
                ),
                ValueError,
            ),
            # 62
            (
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="none",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=True,
                    pad_fill=Validity.VALID,
                    boundary_condition=None,
                ),
                ValueError,
            ),
            # 63
            (
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="none",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=True,
                    pad_fill=Validity.VALID,
                    boundary_condition="reflect",
                ),
                ValueError,
            ),
            # ------------------------------------------------------------------
            # PAD = YES - MASK_KIND = "safe_region"
            # ------------------------------------------------------------------
            # pad = yes | mask_kind = "safe_region" | safe_region = "none"
            # 64
            (
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=None,
                    needs_mask_alloc=False,
                    pad_fill=None,
                    boundary_condition=None,
                ),
                ValueError,
            ),
            # 65
            (
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=None,
                    needs_mask_alloc=False,
                    pad_fill=None,
                    boundary_condition="reflect",
                ),
                ValueError,
            ),
            # 66
            (
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=None,
                    needs_mask_alloc=False,
                    pad_fill=Validity.VALID,
                    boundary_condition=None,
                ),
                ValueError,
            ),
            # 67
            (
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=None,
                    needs_mask_alloc=False,
                    pad_fill=Validity.VALID,
                    boundary_condition="reflect",
                ),
                ValueError,
            ),
            # 68
            (
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=None,
                    needs_mask_alloc=True,
                    pad_fill=None,
                    boundary_condition=None,
                ),
                ValueError,
            ),
            # 69
            (
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=None,
                    needs_mask_alloc=True,
                    pad_fill=None,
                    boundary_condition="reflect",
                ),
                ValueError,
            ),
            # 70
            (
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=None,
                    needs_mask_alloc=True,
                    pad_fill=Validity.VALID,
                    boundary_condition=None,
                ),
                ValueError,
            ),
            # 71
            (
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=None,
                    needs_mask_alloc=True,
                    pad_fill=Validity.VALID,
                    boundary_condition="reflect",
                ),
                ValueError,
            ),
            # pad = yes | mask_kind = "safe_region" | safe_region != "none"
            # 72
            (
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=False,
                    pad_fill=None,
                    boundary_condition=None,
                ),
                ValueError,
            ),
            # 73
            (
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=False,
                    pad_fill=None,
                    boundary_condition="reflect",
                ),
                ValueError,
            ),
            # 74
            (
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=False,
                    pad_fill=Validity.VALID,
                    boundary_condition=None,
                ),
                None,
            ),
            # 75
            (
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=False,
                    pad_fill=Validity.VALID,
                    boundary_condition="reflect",
                ),
                None,
            ),
            # 76
            (
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=True,
                    pad_fill=None,
                    boundary_condition=None,
                ),
                ValueError,
            ),
            # 77
            (
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=True,
                    pad_fill=None,
                    boundary_condition="reflect",
                ),
                ValueError,
            ),
            # 78
            (
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=True,
                    pad_fill=Validity.VALID,
                    boundary_condition=None,
                ),
                None,
            ),
            # 79
            (
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=True,
                    pad_fill=Validity.VALID,
                    boundary_condition="reflect",
                ),
                ValueError,
            ),
            # ------------------------------------------------------------------
            # PAD = YES - MASK_KIND = "binary"
            # ------------------------------------------------------------------
            # pad = yes | mask_kind = "binary" | safe_region = "none"
            # 80
            (
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="binary",
                    safe_region=None,
                    needs_mask_alloc=False,
                    pad_fill=None,
                    boundary_condition=None,
                ),
                ValueError,
            ),
            # 81
            (
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="binary",
                    safe_region=None,
                    needs_mask_alloc=False,
                    pad_fill=None,
                    boundary_condition="reflect",
                ),
                ValueError,
            ),
            # 82
            (
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="binary",
                    safe_region=None,
                    needs_mask_alloc=False,
                    pad_fill=Validity.VALID,
                    boundary_condition=None,
                ),
                None,
            ),
            # 83
            (
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="binary",
                    safe_region=None,
                    needs_mask_alloc=False,
                    pad_fill=Validity.VALID,
                    boundary_condition="reflect",
                ),
                None,
            ),
            # 84
            (
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="binary",
                    safe_region=None,
                    needs_mask_alloc=True,
                    pad_fill=None,
                    boundary_condition=None,
                ),
                ValueError,
            ),
            # 85
            (
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="binary",
                    safe_region=None,
                    needs_mask_alloc=True,
                    pad_fill=None,
                    boundary_condition="reflect",
                ),
                ValueError,
            ),
            # 86
            (
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="binary",
                    safe_region=None,
                    needs_mask_alloc=True,
                    pad_fill=Validity.VALID,
                    boundary_condition=None,
                ),
                ValueError,
            ),
            # 87
            (
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="binary",
                    safe_region=None,
                    needs_mask_alloc=True,
                    pad_fill=Validity.VALID,
                    boundary_condition="reflect",
                ),
                ValueError,
            ),
            # pad = yes | mask_kind = "binary" | safe_region != "none"
            # 88
            (
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="binary",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=False,
                    pad_fill=None,
                    boundary_condition=None,
                ),
                ValueError,
            ),
            # 89
            (
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="binary",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=False,
                    pad_fill=None,
                    boundary_condition="reflect",
                ),
                ValueError,
            ),
            # 90
            (
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="binary",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=False,
                    pad_fill=Validity.VALID,
                    boundary_condition=None,
                ),
                ValueError,
            ),
            # 91
            (
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="binary",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=False,
                    pad_fill=Validity.VALID,
                    boundary_condition="reflect",
                ),
                ValueError,
            ),
            # 92
            (
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="binary",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=True,
                    pad_fill=None,
                    boundary_condition=None,
                ),
                ValueError,
            ),
            # 93
            (
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="binary",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=True,
                    pad_fill=None,
                    boundary_condition="reflect",
                ),
                ValueError,
            ),
            # 94
            (
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="binary",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=True,
                    pad_fill=Validity.VALID,
                    boundary_condition=None,
                ),
                ValueError,
            ),
            # 95
            (
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="binary",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=True,
                    pad_fill=Validity.VALID,
                    boundary_condition="reflect",
                ),
                ValueError,
            ),
        ],
    )
    def test_check_mask_strategy(
        self,
        pad: np.ndarray,
        strategy: ResamplingMaskStrategy,
        expected,
    ):
        """Test mask strategy application on mask"""
        if isinstance(expected, type) and issubclass(expected, Exception):
            with pytest.raises(expected):
                check_mask_strategy(pad, strategy)
        else:
            check_mask_strategy(pad, strategy)

    @pytest.mark.parametrize(
        "array_in_shape",
        # parametrize input shape with the 2d and 3d shape style targeting
        # the same 2d shape
        [
            (1, 2, 3),
            (3, 2, 3),
            (2, 3),
        ],
    )
    @pytest.mark.parametrize(
        "array_in_mask, pad, strategy, expected",
        [
            # ------------------------------------------------------------------
            # PAD = NO - MASK_KIND = "none"
            # ------------------------------------------------------------------
            # pad = no | mask_kind = "none" | safe_region = "none"
            # 0
            (
                None,  # array_in_mask
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="none",
                    safe_region=None,
                    needs_mask_alloc=False,
                    pad_fill=None,
                    boundary_condition=None,
                ),
                None,
            ),
            # 1
            (
                np.asarray(
                    [
                        [1, 1, 1],
                        [1, 1, 1],
                    ]
                ),
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="none",
                    safe_region=None,
                    needs_mask_alloc=False,
                    pad_fill=None,
                    boundary_condition=None,
                ),
                None,
            ),
            # ------------------------------------------------------------------
            # PAD = NO - MASK_KIND = "safe_region"
            # ------------------------------------------------------------------
            # pad = no | mask_kind = "safe_region" | safe_region != "none"
            # 2
            (
                None,  # array_in_mask,
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=((0, 1), (0, 2)),
                    needs_mask_alloc=False,
                    pad_fill=None,
                    boundary_condition=None,
                ),
                ValueError,
            ),
            # 3
            (
                np.asarray(
                    [
                        [1, 1, 1],
                        [1, 1, 1],
                    ]
                ),
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=((0, 1), (0, 2)),
                    needs_mask_alloc=False,
                    pad_fill=None,
                    boundary_condition=None,
                ),
                np.asarray(
                    [
                        [1, 1, 1],
                        [1, 1, 1],
                    ]
                ),
            ),
            # 4
            (
                None,
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=True,
                    pad_fill=Validity.INVALID,
                    boundary_condition=None,
                ),
                np.asarray(
                    [
                        [1, 1, 0],
                        [1, 1, 0],
                    ]
                ),
            ),
            # 5
            (
                None,
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=True,
                    pad_fill=Validity.VALID,
                    boundary_condition=None,
                ),
                np.asarray(
                    [
                        [1, 1, 1],
                        [1, 1, 1],
                    ]
                ),
            ),
            # 6
            (
                np.asarray(
                    [
                        [1, 1, 1],
                        [1, 1, 0],
                    ]
                ),
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=False,
                    pad_fill=None,
                    boundary_condition=None,
                ),
                np.asarray(
                    [
                        [1, 1, 1],
                        [1, 1, 0],
                    ]
                ),
            ),
            # ------------------------------------------------------------------
            # PAD = NO - MASK_KIND = "binary"
            # ------------------------------------------------------------------
            # pad = no | mask_kind = "binary" | safe_region = "none"
            # 7
            (
                np.asarray(
                    [
                        [1, 1, 1],
                        [1, 1, 0],
                    ]
                ),
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="binary",
                    safe_region=None,
                    needs_mask_alloc=False,
                    pad_fill=None,
                    boundary_condition=None,
                ),
                np.asarray(
                    [
                        [1, 1, 1],
                        [1, 1, 0],
                    ]
                ),
            ),
            # 8
            (
                None,
                np.array([[0, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="binary",
                    safe_region=None,
                    needs_mask_alloc=False,
                    pad_fill=None,
                    boundary_condition=None,
                ),
                ValueError,
            ),
            # ------------------------------------------------------------------
            # PAD = YES - MASK_KIND = "none"
            # ------------------------------------------------------------------
            # pad = yes | mask_kind = "none" | safe_region = "none"
            # 9
            (
                None,
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="none",
                    safe_region=None,
                    needs_mask_alloc=False,
                    pad_fill=None,
                    boundary_condition=None,
                ),
                None,
            ),
            # 10
            (
                np.asarray(
                    [
                        [1, 1, 1],
                        [1, 1, 1],
                    ]
                ),
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="none",
                    safe_region=None,
                    needs_mask_alloc=False,
                    pad_fill=None,
                    boundary_condition=None,
                ),
                None,
            ),
            # ------------------------------------------------------------------
            # PAD = YES - MASK_KIND = "safe_region"
            # ------------------------------------------------------------------
            # pad = yes | mask_kind = "safe_region" | safe_region = "none"
            # 11
            (
                None,
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=True,
                    pad_fill=Validity.VALID,
                    boundary_condition=None,
                ),
                np.asarray(
                    [
                        [1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1],
                    ]
                ),
            ),
            # 12
            (
                None,
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=True,
                    pad_fill=Validity.INVALID,
                    boundary_condition=None,
                ),
                np.asarray(
                    [
                        [1, 1, 0],
                        [1, 1, 0],
                        [0, 0, 0],
                    ]
                ),
            ),
            # 13
            (
                None,
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=((0, 1), (1, 2)),
                    needs_mask_alloc=True,
                    pad_fill=Validity.INVALID,
                    boundary_condition=None,
                ),
                np.asarray(
                    [
                        [0, 1, 1],
                        [0, 1, 1],
                        [0, 0, 0],
                    ]
                ),
            ),
            # 14 - check the application does not change the input mask
            (
                np.asarray(
                    [
                        [1, 1, 1],
                        [1, 1, 1],
                    ]
                ),
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=((0, 0), (0, 0)),
                    needs_mask_alloc=False,
                    pad_fill=Validity.INVALID,
                    boundary_condition=None,
                ),
                np.asarray(
                    [
                        [0, 0, 0],
                        [1, 1, 1],
                        [1, 1, 1],
                    ]
                ),
            ),
            # 15
            (
                np.asarray(
                    [
                        [1, 1, 0],
                        [1, 1, 0],
                    ]
                ),
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=((1, 2), (0, 1)),
                    needs_mask_alloc=False,
                    pad_fill=Validity.VALID,
                    boundary_condition="reflect",
                ),
                np.asarray(
                    [
                        [1, 1, 0],
                        [1, 1, 0],
                        [1, 1, 0],
                    ]
                ),
            ),
            # 16
            (
                None,
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=True,
                    pad_fill=Validity.INVALID,
                    boundary_condition=None,
                ),
                np.asarray(
                    [
                        [1, 1, 0],
                        [1, 1, 0],
                        [0, 0, 0],
                    ]
                ),
            ),
            # 17
            (
                None,
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="safe_region",
                    safe_region=((0, 1), (0, 1)),
                    needs_mask_alloc=True,
                    pad_fill=Validity.VALID,
                    boundary_condition=None,
                ),
                np.asarray(
                    [
                        [1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1],
                    ]
                ),
            ),
            # ------------------------------------------------------------------
            # PAD = YES - MASK_KIND = "binary"
            # ------------------------------------------------------------------
            # pad = yes | mask_kind = "binary" | safe_region = "none"
            # 18
            (
                np.asarray(
                    [
                        [1, 0, 1],
                        [1, 1, 1],
                    ]
                ),
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="binary",
                    safe_region=None,
                    needs_mask_alloc=False,
                    pad_fill=Validity.VALID,
                    boundary_condition=None,
                ),
                np.asarray(
                    [
                        [1, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1],
                    ]
                ),
            ),
            # 19
            (
                np.asarray(
                    [
                        [1, 0, 1],
                        [1, 1, 1],
                    ]
                ),
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="binary",
                    safe_region=None,
                    needs_mask_alloc=False,
                    pad_fill=Validity.VALID,
                    boundary_condition="reflect",
                ),
                np.asarray(
                    [
                        [1, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1],
                    ]
                ),
            ),
            # 20
            (
                np.asarray(
                    [
                        [1, 0, 1],
                        [1, 1, 1],
                    ]
                ),
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="binary",
                    safe_region=None,
                    needs_mask_alloc=False,
                    pad_fill=Validity.VALID,
                    boundary_condition="symmetric",
                ),
                np.asarray(
                    [
                        [1, 0, 1],
                        [1, 0, 1],
                        [1, 1, 1],
                    ]
                ),
            ),
            # 21
            (
                np.asarray(
                    [
                        [1, 0, 1],
                        [1, 1, 1],
                    ]
                ),
                np.array([[1, 0], [0, 0]]),  # pad
                ResamplingMaskStrategy(
                    mask_kind="binary",
                    safe_region=None,
                    needs_mask_alloc=False,
                    pad_fill=Validity.INVALID,
                    boundary_condition=None,
                ),
                np.asarray(
                    [
                        [0, 0, 0],
                        [1, 0, 1],
                        [1, 1, 1],
                    ]
                ),
            ),
        ],
    )
    def test_apply_mask_strategy(
        self,
        array_in_mask: Optional[np.ndarray],
        pad: np.ndarray,
        array_in_shape: Tuple[int, ...],
        strategy: ResamplingMaskStrategy,
        expected,
    ):
        """Test mask strategy application on mask"""
        if isinstance(expected, type) and issubclass(expected, Exception):
            with pytest.raises(expected):
                mask = apply_mask_strategy(
                    array_in_mask,
                    pad,
                    array_in_shape,
                    strategy,
                )
        else:
            mask = apply_mask_strategy(
                array_in_mask,
                pad,
                array_in_shape,
                strategy,
            )
            assert optional_array_equal(mask, expected)

    # ---------------------------------------------------------------------------
    # Test array_grid_resampling
    # ---------------------------------------------------------------------------
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

    @pytest.mark.parametrize(
        "array, grid, interp, mask, check_boundaries, testing_decimal",
        [
            (
                ARRAY_IN_001_ARRAY,
                GRID_IN_001_01_GRID,
                "nearest",
                MASK_IN_001_ARRAY_01,
                True,
                6,
            ),
        ],
    )
    def test_grid_resampling_standalone_no_idendity_1_1_full_invalid_grid(
        self,
        array,
        grid,
        interp,
        mask,
        check_boundaries,
        testing_decimal,
    ):
        """Test idendity resampling with a full resolute grid"""
        nodata_out = -10
        standalone = False
        boundary_condition = None

        grid_mask = np.zeros(grid[0].shape, dtype=np.uint8)
        grid_mask_valid_value = 1
        grid_nodata = None

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
            grid_nodata=grid_nodata,
            array_out_mask=mask is not None,
            check_boundaries=check_boundaries,
            standalone=standalone,
            boundary_condition=boundary_condition,
        )
        np.testing.assert_array_almost_equal(
            np.full(grid[0].shape, nodata_out, dtype=np.float64),
            array_out,
            decimal=testing_decimal,
        )
        np.testing.assert_array_almost_equal(
            np.full(grid[0].shape, MASKED_VALUE, dtype=np.uint8),
            mask_out,
            decimal=testing_decimal,
        )
