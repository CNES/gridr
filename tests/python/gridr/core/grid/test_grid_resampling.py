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
import copy
from functools import partialmethod
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
from gridr.core.utils.array_window import window_indices

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

        # Also check that if boundary_condition is None, trust_padding has no
        # effect on the strategy
        if boundary_condition is None:
            # compute with the negation of trust_padding
            strategy = resolve_mask_strategy(
                interp,
                pad,
                array_in_mask,
                boundary_condition,
                ~trust_padding,
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


class TestGridResamplingMonoPointGrid:

    NODATA_OUT = -9999.0

    # -- helper function for general mono point grid case --
    def _mono_point_grid_generate_input(
        self,
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
        bsplines_kwargs=None,
    ):
        """
        interp=interp,
        array_in=image,
        grid_row=np.asarray(grid_row),
        grid_col=np.asarray(grid_col),
        grid_resolution=resolution,
        nodata_out=nodata_out,
        array_in_origin=array_in_origin,
        win=win,
        array_in_mask=array_in_mask,
        array_in_mask_safe_win=array_in_mask_safe_win,
        grid_mask=grid_mask,
        grid_mask_valid_value=grid_mask_valid_value,
        grid_nodata=grid_nodata,
        array_out_mask=True,
        check_boundaries=check_boundaries,
        standalone=standalone,
        boundary_condition=boundary_condition,
        trust_padding=trust_padding,
        """
        if len(shape_array_in) == 3:
            _, array_in_row, array_in_col = shape_array_in
        else:
            array_in_row, array_in_col = shape_array_in
        kwargs = {}
        # -- interp --
        if "bspline" in interp:
            if bsplines_kwargs is None:
                bsplines_kwargs = {"epsilon": 1e-2, "mask_influence_threshold": 1}
            interp = get_interpolator(interp, **bsplines_kwargs)
        else:
            interp = get_interpolator(interp)
        interp.initialize()
        kwargs["interp"] = interp
        # -- array_in --
        if array_in_cst_value is not None:
            kwargs["array_in"] = (
                np.ones(np.prod(shape_array_in), dtype=np.float64).reshape(shape_array_in)
                * array_in_cst_value
            )
        else:
            kwargs["array_in"] = np.arange(np.prod(shape_array_in), dtype=np.float64).reshape(
                shape_array_in
            )
        # -- array_in_origin --
        kwargs["array_in_origin"] = array_in_origin
        # -- resolution --
        kwargs["grid_resolution"] = resolution
        # -- array_in_mask
        kwargs["array_in_mask"] = None
        kwargs["array_in_mask_safe_win"] = None
        if use_array_in_mask:
            array_in_mask = np.ones(kwargs["array_in"].shape, dtype=np.uint8, order="C")
            if set_array_in_mask_invalid:
                array_in_mask[0, 0] = 0
                array_in_mask[17, 27] = 0
            kwargs["array_in_mask"] = array_in_mask
            if set_array_in_mask_safe_window:
                if set_array_in_mask_invalid:
                    kwargs["array_in_mask_safe_win"] = np.asarray([(1, 16), (1, 26)])
                else:
                    kwargs["array_in_mask_safe_win"] = np.asarray(
                        [(0, array_in_row - 1), (0, array_in_col - 1)]
                    )

        def get_scalar_grid(x, y):
            grid_row = np.asarray(
                [
                    [
                        y,
                    ],
                ]
            )
            grid_col = np.asarray(
                [
                    [
                        x,
                    ],
                ]
            )
            return np.atleast_2d(grid_row), np.atleast_2d(grid_col)

        grid_row, grid_col = get_scalar_grid(x=col_target, y=row_target)
        kwargs["grid_row"] = grid_row
        kwargs["grid_col"] = grid_col

        kwargs["array_out_mask"] = array_out_mask

        grid_mask = None
        if use_grid_mask:
            grid_mask = np.full(np.asarray(grid_row).shape, grid_mask_valid_value, dtype=np.uint8)
            # make last column invalid
            grid_mask[:, -1] = grid_mask_valid_value - 1
        kwargs["grid_mask"] = grid_mask
        kwargs["grid_mask_valid_value"] = grid_mask_valid_value
        kwargs["grid_nodata"] = None

        kwargs["nodata_out"] = nodata_out
        kwargs["win"] = win
        if kwargs["win"] is not None:
            kwargs["win"] = np.asarray(kwargs["win"])

        # interp = get_interpolator("bspline3", **{'epsilon': 1e-2, 'mask_influence_threshold': 1})
        interp = get_interpolator("linear")
        interp = get_interpolator("cubic")
        interp.initialize()

        kwargs["check_boundaries"] = check_boundaries
        kwargs["standalone"] = use_standalone
        kwargs["boundary_condition"] = boundary_condition
        kwargs["trust_padding"] = trust_padding

        return kwargs

    # -- Helper for nominal cases --
    _mono_point_grid_generate_input__partial_nominal_case = partialmethod(
        _mono_point_grid_generate_input,
        array_in_cst_value=None,
        shape_array_in=(20, 30),
        resolution=(1, 1),
        set_array_in_mask_safe_window=False,
        array_out_mask=True,
        use_standalone=True,
        nodata_out=NODATA_OUT,
        array_in_origin=(0.0, 0.0),
        win=None,
        use_grid_mask=False,
        grid_mask_valid_value=1,
        check_boundaries=True,
    )

    def _mono_point_grid_generate_input__partial_nominal_case_build_kwargs(self, args):
        return dict(
            zip(
                (
                    "row_target",
                    "col_target",
                    "use_array_in_mask",
                    "set_array_in_mask_invalid",
                    "set_array_in_mask_safe_window",
                    "interp",
                    "boundary_condition",
                    "trust_padding",
                    "bsplines_kwargs",
                ),
                args,
                strict=True,
            )
        )

    # Default BSpline arguments - here we deactivate the mask dilatation during prefiltering
    DFLT_BSPLINES_ARGS = {"epsilon": 1e-2, "mask_influence_threshold": 1}
    # Here we activate the mask dilatation during prefiltering
    PREFIL_MSK_DILATE_BSPLINES_ARGS = {"epsilon": 1e-2, "mask_influence_threshold": 0.001}

    @pytest.mark.parametrize(
        "args, expected_resampled_array, expected_resampled_mask",
        [
            # -- Tests idendity --
            # -- Test identity at center --
            ((10.0, 15.0, False, False, False, "nearest", "reflect", True, {}), [[315.0]], [[1]]),
            ((10.0, 15.0, False, False, False, "linear", "reflect", True, {}), [[315.0]], [[1]]),
            ((10.0, 15.0, False, False, False, "cubic", "reflect", True, {}), [[315.0]], [[1]]),
            (
                (10.0, 15.0, False, False, False, "bspline3", "reflect", True, DFLT_BSPLINES_ARGS),
                [[315.0]],
                [[1]],
            ),
            # -- Test idendity at Upper Left Corner --
            #   pad = "reflect" && trust_padding = "True"
            ((0.0, 0.0, False, False, False, "nearest", "reflect", True, {}), [[0.0]], [[1]]),
            ((0.0, 0.0, False, False, False, "linear", "reflect", True, {}), [[0.0]], [[1]]),
            ((0.0, 0.0, False, False, False, "cubic", "reflect", True, {}), [[0.0]], [[1]]),
            (
                (0.0, 0.0, False, False, False, "bspline3", "reflect", True, DFLT_BSPLINES_ARGS),
                [[0.0]],
                [[1]],
            ),
            # -- Test idendity at Upper Left Corner -
            #   pad = "reflect" && trust_padding = "False"
            ((0.0, 0.0, False, False, False, "nearest", "reflect", False, {}), [[0.0]], [[1]]),
            ((0.0, 0.0, False, False, False, "linear", "reflect", False, {}), [[0.0]], [[1]]),
            ((0.0, 0.0, False, False, False, "cubic", "reflect", False, {}), [[0.0]], [[1]]),
            # There is a BSpline specificity here as the prefiltering
            # requires untrusted area and the idendity transform requires
            # to convolve the prefiltered image.
            (
                (0.0, 0.0, False, False, False, "bspline3", "reflect", False, DFLT_BSPLINES_ARGS),
                [[NODATA_OUT]],
                [[0]],
            ),
            # -- Test idendity at Upper Left Corner -
            #   pad = None && trust_padding = "False"
            ((0.0, 0.0, False, False, False, "nearest", None, False, {}), [[0.0]], [[1]]),
            ((0.0, 0.0, False, False, False, "linear", None, False, {}), [[0.0]], [[1]]),
            ((0.0, 0.0, False, False, False, "cubic", None, False, {}), [[0.0]], [[1]]),
            # There is a BSpline specificity here as the prefiltering
            # requires untrusted area and the idendity transform requires
            # to convolve the prefiltered image.
            (
                (0.0, 0.0, False, False, False, "bspline3", None, False, DFLT_BSPLINES_ARGS),
                [[NODATA_OUT]],
                [[0]],
            ),
            # -- Test idendity at Upper Left Corner -
            #   pad = None && trust_padding = "True"
            ((0.0, 0.0, False, False, False, "nearest", None, True, {}), [[0.0]], [[1]]),
            ((0.0, 0.0, False, False, False, "linear", None, True, {}), [[0.0]], [[1]]),
            ((0.0, 0.0, False, False, False, "cubic", None, True, {}), [[0.0]], [[1]]),
            # There is a BSpline specificity here as the prefiltering
            # requires untrusted area and the idendity transform requires
            # to convolve the prefiltered image.
            # Trust padding is ignore with boundary condition at None - we must use "constant"
            (
                (0.0, 0.0, False, False, False, "bspline3", None, True, DFLT_BSPLINES_ARGS),
                [[NODATA_OUT]],
                [[0]],
            ),
            (
                (0.0, 0.0, False, False, False, "bspline3", "constant", True, DFLT_BSPLINES_ARGS),
                [[0.0]],
                [[1]],
            ),
            # -- Test idendity at Bottom Right Corner -
            #   pad = None && trust_padding = "True"
            ((19.0, 29.0, False, False, False, "nearest", None, True, {}), [[599.0]], [[1]]),
            ((19.0, 29.0, False, False, False, "linear", None, True, {}), [[599.0]], [[1]]),
            ((19.0, 29.0, False, False, False, "cubic", None, True, {}), [[599.0]], [[1]]),
            # Trust padding is ignore with boundary condition at None - we must use "constant"
            (
                (19.0, 29.0, False, False, False, "bspline3", None, True, DFLT_BSPLINES_ARGS),
                [[NODATA_OUT]],
                [[0]],
            ),
            (
                (19.0, 29.0, False, False, False, "bspline3", "constant", True, DFLT_BSPLINES_ARGS),
                [[599.0]],
                [[1]],
            ),
            # -- Tests idendity with raster mask - invalid at (0, 0) and (17, 27) when enabled --
            # ---- full valid mask ----
            ((0.0, 0.0, True, False, False, "nearest", "reflect", True, {}), [[0.0]], [[1]]),
            ((0.0, 0.0, True, False, False, "linear", "reflect", True, {}), [[0.0]], [[1]]),
            ((0.0, 0.0, True, False, False, "cubic", "reflect", True, {}), [[0.0]], [[1]]),
            (
                (0.0, 0.0, True, False, False, "bspline3", "reflect", True, DFLT_BSPLINES_ARGS),
                [[0.0]],
                [[1]],
            ),
            # ---- invalid at (0,0) only - target (0,0) ----
            ((0.0, 0.0, True, True, False, "nearest", "reflect", True, {}), [[NODATA_OUT]], [[0]]),
            ((0.0, 0.0, True, True, False, "linear", "reflect", True, {}), [[NODATA_OUT]], [[0]]),
            ((0.0, 0.0, True, True, False, "cubic", "reflect", True, {}), [[NODATA_OUT]], [[0]]),
            (
                (0.0, 0.0, True, True, False, "bspline3", "reflect", True, DFLT_BSPLINES_ARGS),
                [[NODATA_OUT]],
                [[0]],
            ),
            # ---- invalid at (0,0) only - target (1,1) ----
            ((1.0, 1.0, True, True, False, "nearest", "reflect", True, {}), [[31.0]], [[1]]),
            ((1.0, 1.0, True, True, False, "linear", "reflect", True, {}), [[31.0]], [[1]]),
            ((1.0, 1.0, True, True, False, "cubic", "reflect", True, {}), [[31.0]], [[1]]),
            # Bspline results in nodata because of the required convolution after prefiltering
            # (even for idendity)
            # Here the invalid point after prefiltering lies within the convolution stencil
            (
                (1.0, 1.0, True, True, False, "bspline3", "reflect", True, DFLT_BSPLINES_ARGS),
                [[NODATA_OUT]],
                [[0]],
            ),
            # If we target (2., 2.), the invalid data do not lie within the convnolution stencil
            # Note default bspline args deactivate mask dilatation during prefiltering (value 1)
            (
                (2.0, 2.0, True, True, False, "bspline3", "reflect", True, DFLT_BSPLINES_ARGS),
                [[62.0]],
                [[1]],
            ),
            # If we activate the mask diltation it results in nodata as output
            (
                (
                    2.0,
                    2.0,
                    True,
                    True,
                    False,
                    "bspline3",
                    "reflect",
                    True,
                    {"epsilon": 0.01, "mask_influence_threshold": 0.001},
                ),
                [[NODATA_OUT]],
                [[0]],
            ),
            # Tests shift on col (with valid / invalid mask)
            # -- Tests shift on col / idendity on row - invalid at (0, 0) and (17, 27) if enabled --
            # ---- full valid mask ----
            # ------ nearest ------
            # -------- within domain --------
            ((0.0, 0.49, True, False, False, "nearest", "reflect", True, {}), [[0.000000]], [[1]]),
            ((0.0, 0.49, True, False, False, "nearest", "reflect", False, {}), [[0.000000]], [[1]]),
            ((0.0, 0.49, True, False, False, "nearest", None, False, {}), [[0.000000]], [[1]]),
            ((0.0, 0.5, True, False, False, "nearest", "reflect", True, {}), [[1.000000]], [[1]]),
            ((0.0, 0.51, True, False, False, "nearest", "reflect", True, {}), [[1.000000]], [[1]]),
            (
                (0.0, 29.49, True, False, False, "nearest", "reflect", True, {}),
                [[29.000000]],
                [[1]],
            ),
            # -------- outside of domain --------
            ((0.0, -0.49, True, False, False, "nearest", "reflect", True, {}), [[0.0]], [[1]]),
            ((0.0, -0.49, True, False, False, "nearest", "reflect", False, {}), [[0.0]], [[1]]),
            ((0.0, -0.49, True, False, False, "nearest", None, False, {}), [[0.0]], [[1]]),
            ((0.0, -0.5, True, False, False, "nearest", "reflect", True, {}), [[0.0]], [[1]]),
            ((0.0, -0.5, True, False, False, "nearest", "reflect", False, {}), [[0.0]], [[1]]),
            (
                (0.0, -0.51, True, False, False, "nearest", "reflect", True, {}),
                [[NODATA_OUT]],
                [[0]],
            ),
            (
                (0.0, -0.51, True, False, False, "nearest", "reflect", False, {}),
                [[NODATA_OUT]],
                [[0]],
            ),
            (
                (0.0, 29.5, True, False, False, "nearest", "reflect", True, {}),
                [[NODATA_OUT]],
                [[0]],
            ),
            # ------ linear ------
            # -------- target within domain - interpolation stencil outside the domain --------
            ((0.0, 0.1, True, False, False, "linear", "reflect", True, {}), [[0.1]], [[1]]),
            ((0.0, 0.1, True, False, False, "linear", "reflect", False, {}), [[0.1]], [[1]]),
            ((0.0, 0.5, True, False, False, "linear", "reflect", True, {}), [[0.5]], [[1]]),
            ((0.0, 0.8, True, False, False, "linear", "reflect", False, {}), [[0.8]], [[1]]),
            ((0.0, 28.99, True, False, False, "linear", "reflect", True, {}), [[28.990000]], [[1]]),
            (
                (0.0, 28.99, True, False, False, "linear", "reflect", False, {}),
                [[28.990000]],
                [[1]],
            ),
            ((0.0, 29.0, True, False, False, "linear", "reflect", False, {}), [[29.0]], [[1]]),
            # -------- outside of domain  - interpolation stencil outside of the domain --------
            ((0.0, -0.1, True, False, False, "linear", "reflect", True, {}), [[NODATA_OUT]], [[0]]),
            (
                (0.0, -0.1, True, False, False, "linear", "reflect", False, {}),
                [[NODATA_OUT]],
                [[0]],
            ),
            (
                (0.0, 29.01, True, False, False, "linear", "reflect", True, {}),
                [[NODATA_OUT]],
                [[0]],
            ),
            # ------ cubic ------
            # -------- target within domain - interpolation stencil outside the domain --------
            ((0.0, 0.1, True, False, False, "cubic", "reflect", True, {}), [[0.019000]], [[1]]),
            ((0.0, 0.1, True, False, False, "cubic", "reflect", False, {}), [[NODATA_OUT]], [[0]]),
            ((0.0, 0.1, True, False, False, "cubic", "constant", True, {}), [[0.059500]], [[1]]),
            (
                (0.0, 0.999, True, False, False, "cubic", "reflect", False, {}),
                [[NODATA_OUT]],
                [[0]],
            ),
            ((0.0, 28.999, True, False, False, "cubic", "reflect", True, {}), [[28.999998]], [[1]]),
            (
                (0.0, 28.999, True, False, False, "cubic", "reflect", False, {}),
                [[NODATA_OUT]],
                [[0]],
            ),
            # -------- target within domain - interpolation stencil within the domain --------
            ((0.0, 1.001, True, False, False, "cubic", "reflect", False, {}), [[1.001000]], [[1]]),
            ((0.0, 1.1, True, False, False, "cubic", "reflect", True, {}), [[1.100000]], [[1]]),
            ((0.0, 1.1, True, False, False, "cubic", "reflect", False, {}), [[1.100000]], [[1]]),
            # -------- outside of domain  - interpolation stencil outside of the domain --------
            ((0.0, -0.1, True, False, False, "cubic", "reflect", True, {}), [[NODATA_OUT]], [[0]]),
            ((0.0, -0.1, True, False, False, "cubic", "reflect", False, {}), [[NODATA_OUT]], [[0]]),
            # ------ bspline3 ------
            # -------- target within domain - interpolation stencil outside the domain --------
            (
                (0.0, 0.001, True, False, False, "bspline3", "reflect", True, DFLT_BSPLINES_ARGS),
                [[0.000001]],
                [[1]],
            ),
            (
                (0.0, 0.999, True, False, False, "bspline3", "reflect", True, DFLT_BSPLINES_ARGS),
                [[0.998731]],
                [[1]],
            ),
            (
                (1.0, 0.999, True, False, False, "bspline3", "reflect", True, DFLT_BSPLINES_ARGS),
                [[30.998731]],
                [[1]],
            ),
            (
                (1.0, 0.999, True, False, False, "bspline3", "reflect", False, DFLT_BSPLINES_ARGS),
                [[NODATA_OUT]],
                [[0]],
            ),
            (
                (0.0, 28.999, True, False, False, "bspline3", "reflect", True, DFLT_BSPLINES_ARGS),
                [[28.999997]],
                [[1]],
            ),
            (
                (0.0, 28.999, True, False, False, "bspline3", "reflect", False, DFLT_BSPLINES_ARGS),
                [[NODATA_OUT]],
                [[0]],
            ),
            # -------- target within domain - interpolation stencil within the domain --------
            (
                (0.0, 1.001, True, False, False, "bspline3", "reflect", True, DFLT_BSPLINES_ARGS),
                [[1.001267]],
                [[1]],
            ),
            (
                (0.0, 1.001, True, False, False, "bspline3", "reflect", False, DFLT_BSPLINES_ARGS),
                [[NODATA_OUT]],
                [[0]],
            ),
            (
                (1.0, 1.001, True, False, False, "bspline3", "reflect", True, DFLT_BSPLINES_ARGS),
                [[31.001265]],
                [[1]],
            ),
            (
                (
                    1.001,
                    1.0,
                    True,
                    False,
                    False,
                    "bspline3",
                    "reflect",
                    False,
                    PREFIL_MSK_DILATE_BSPLINES_ARGS,
                ),
                [[NODATA_OUT]],
                [[0]],
            ),
            # -------- outside of domain  - interpolation stencil outside of the domain --------
            (
                (0.0, -0.001, True, False, False, "bspline3", "reflect", True, DFLT_BSPLINES_ARGS),
                [[NODATA_OUT]],
                [[1]],
            ),
            (
                (0.0, -0.001, True, False, False, "bspline3", "reflect", False, DFLT_BSPLINES_ARGS),
                [[NODATA_OUT]],
                [[1]],
            ),
            # -- Tests shift on row / idendity on col - invalid at (0, 0) and (17, 27) if enabled --
            # ---- full valid mask ----
            # ------ nearest ------
            # -------- within domain --------
            ((0.49, 0.0, True, False, False, "nearest", "reflect", True, {}), [[0.0]], [[1]]),
            ((0.49, 0.0, True, False, False, "nearest", "reflect", False, {}), [[0.0]], [[1]]),
            ((0.49, 0.0, True, False, False, "nearest", None, False, {}), [[0.0]], [[1]]),
            ((0.5, 0.0, True, False, False, "nearest", "reflect", True, {}), [[30.0]], [[1]]),
            ((0.51, 0.0, True, False, False, "nearest", "reflect", True, {}), [[30.0]], [[1]]),
            (
                (19.49, 0.0, True, False, False, "nearest", "reflect", True, {}),
                [[570.000000]],
                [[1]],
            ),
            # -------- outside of domain --------
            ((-0.49, 0.0, True, False, False, "nearest", "reflect", True, {}), [[0.0]], [[1]]),
            ((-0.49, 0.0, True, False, False, "nearest", "reflect", False, {}), [[0.0]], [[1]]),
            ((-0.49, 0.0, True, False, False, "nearest", None, False, {}), [[0.0]], [[1]]),
            ((-0.5, 0.0, True, False, False, "nearest", "reflect", True, {}), [[0.0]], [[1]]),
            ((-0.5, 0.0, True, False, False, "nearest", "reflect", False, {}), [[0.0]], [[1]]),
            (
                (-0.51, 0.0, True, False, False, "nearest", "reflect", True, {}),
                [[NODATA_OUT]],
                [[0]],
            ),
            (
                (-0.51, 0.0, True, False, False, "nearest", "reflect", False, {}),
                [[NODATA_OUT]],
                [[0]],
            ),
            (
                (19.5, 0.0, True, False, False, "nearest", "reflect", True, {}),
                [[NODATA_OUT]],
                [[0]],
            ),
            # ------ linear ------
            # -------- target within domain - interpolation stencil outside the domain --------
            ((0.1, 0.0, True, False, False, "linear", "reflect", True, {}), [[3.0]], [[1]]),
            ((0.1, 0.0, True, False, False, "linear", "reflect", False, {}), [[3.0]], [[1]]),
            ((0.5, 0.0, True, False, False, "linear", "reflect", True, {}), [[15.0]], [[1]]),
            ((0.8, 0.0, True, False, False, "linear", "reflect", True, {}), [[24.0]], [[1]]),
            (
                (18.99, 0.0, True, False, False, "linear", "reflect", True, {}),
                [[569.700000]],
                [[1]],
            ),
            (
                (18.99, 0.0, True, False, False, "linear", "reflect", False, {}),
                [[569.700000]],
                [[1]],
            ),
            ((19.0, 0.0, True, False, False, "linear", "reflect", True, {}), [[570.000000]], [[1]]),
            # -------- outside of domain  - interpolation stencil outside of the domain --------
            ((-0.1, 0.0, True, False, False, "linear", "reflect", True, {}), [[NODATA_OUT]], [[0]]),
            (
                (-0.1, 0.0, True, False, False, "linear", "reflect", False, {}),
                [[NODATA_OUT]],
                [[0]],
            ),
            (
                (19.01, 0.0, True, False, False, "linear", "reflect", True, {}),
                [[NODATA_OUT]],
                [[0]],
            ),
            # ------ cubic ------
            # -------- target within domain - interpolation stencil outside the domain --------
            ((0.1, 0.0, True, False, False, "cubic", "reflect", True, {}), [[0.57]], [[1]]),
            ((0.1, 0.0, True, False, False, "cubic", "reflect", False, {}), [[NODATA_OUT]], [[0]]),
            ((0.1, 0.0, True, False, False, "cubic", "constant", True, {}), [[1.785]], [[1]]),
            (
                (0.999, 0.0, True, False, False, "cubic", "reflect", False, {}),
                [[NODATA_OUT]],
                [[0]],
            ),
            ((18.99, 0.0, True, False, False, "cubic", "reflect", True, {}), [[569.994030]], [[1]]),
            (
                (18.99, 0.0, True, False, False, "cubic", "reflect", False, {}),
                [[NODATA_OUT]],
                [[0]],
            ),
            # -------- target within domain - interpolation stencil within the domain --------
            ((1.001, 0.0, True, False, False, "cubic", "reflect", False, {}), [[30.03]], [[1]]),
            ((1.1, 0.0, True, False, False, "cubic", "reflect", True, {}), [[33.0]], [[1]]),
            ((1.1, 0.0, True, False, False, "cubic", "reflect", False, {}), [[33.0]], [[1]]),
            # -------- outside of domain  - interpolation stencil outside of the domain --------
            ((-0.1, 0.0, True, False, False, "cubic", "reflect", True, {}), [[NODATA_OUT]], [[0]]),
            ((-0.1, 0.0, True, False, False, "cubic", "reflect", False, {}), [[NODATA_OUT]], [[0]]),
            # ------ bspline3 ------
            # -------- target within domain - interpolation stencil outside the domain --------
            (
                (0.001, 0.0, True, False, False, "bspline3", "reflect", True, DFLT_BSPLINES_ARGS),
                [[0.000037]],
                [[1]],
            ),
            (
                (0.999, 0.0, True, False, False, "bspline3", "reflect", True, DFLT_BSPLINES_ARGS),
                [[29.961944]],
                [[1]],
            ),
            (
                (0.999, 1.0, True, False, False, "bspline3", "reflect", True, DFLT_BSPLINES_ARGS),
                [[30.961944]],
                [[1]],
            ),
            (
                (0.999, 1.0, True, False, False, "bspline3", "reflect", False, DFLT_BSPLINES_ARGS),
                [[NODATA_OUT]],
                [[0]],
            ),
            (
                (18.99, 0.0, True, False, False, "bspline3", "reflect", True, DFLT_BSPLINES_ARGS),
                [[569.994618]],
                [[1]],
            ),
            (
                (18.99, 0.0, True, False, False, "bspline3", "reflect", False, DFLT_BSPLINES_ARGS),
                [[NODATA_OUT]],
                [[0]],
            ),
            # -------- target within domain - interpolation stencil within the domain --------
            (
                (1.001, 0.0, True, False, False, "bspline3", "reflect", True, DFLT_BSPLINES_ARGS),
                [[30.038012]],
                [[1]],
            ),
            (
                (1.001, 0.0, True, False, False, "bspline3", "reflect", False, DFLT_BSPLINES_ARGS),
                [[-9999.000000]],
                [[0]],
            ),
            (
                (1.001, 1.0, True, False, False, "bspline3", "reflect", False, DFLT_BSPLINES_ARGS),
                [[31.038012]],
                [[1]],
            ),
            (
                (
                    1.001,
                    1.0,
                    True,
                    False,
                    False,
                    "bspline3",
                    "reflect",
                    False,
                    PREFIL_MSK_DILATE_BSPLINES_ARGS,
                ),
                [[NODATA_OUT]],
                [[0]],
            ),
            # -------- outside of domain  - interpolation stencil outside of the domain --------
            (
                (-0.001, 0.0, True, False, False, "bspline3", "reflect", True, DFLT_BSPLINES_ARGS),
                [[NODATA_OUT]],
                [[1]],
            ),
            (
                (-0.001, 0.0, True, False, False, "bspline3", "reflect", False, DFLT_BSPLINES_ARGS),
                [[NODATA_OUT]],
                [[1]],
            ),
            # ---- invalid mask ----
            # ------ nearest ------
            # -------- target invalid point --------
            ((0.0, 0.0, True, True, False, "nearest", "reflect", True, {}), [[NODATA_OUT]], [[0]]),
            (
                (17.0, 27.0, True, True, False, "nearest", "reflect", True, {}),
                [[NODATA_OUT]],
                [[0]],
            ),
            # -------- target valid point --------
            ((18.0, 27.0, True, True, False, "nearest", "reflect", True, {}), [[567.0]], [[1]]),
            ((16.0, 27.0, True, True, False, "nearest", "reflect", True, {}), [[507]], [[1]]),
            ((17.0, 26.0, True, True, False, "nearest", "reflect", True, {}), [[536.0]], [[1]]),
            ((17.0, 28.0, True, True, False, "nearest", "reflect", True, {}), [[538.0]], [[1]]),
            # ------ linear ------
            # -------- target invalid point --------
            ((0.0, 0.0, True, True, False, "linear", "reflect", True, {}), [[NODATA_OUT]], [[0]]),
            ((17.0, 27.0, True, True, False, "linear", "reflect", True, {}), [[NODATA_OUT]], [[0]]),
            # -------- target neighbor point - weight for invalid data should be 0 --------
            ((18.0, 27.0, True, True, False, "linear", "reflect", True, {}), [[567.0]], [[1]]),
            ((16.0, 27.0, True, True, False, "linear", "reflect", True, {}), [[507]], [[1]]),
            ((17.0, 26.0, True, True, False, "linear", "reflect", True, {}), [[536.0]], [[1]]),
            ((17.0, 28.0, True, True, False, "linear", "reflect", True, {}), [[538.0]], [[1]]),
            # -------- target neighbor point - interpolation required --------
            (
                (17.99, 27.0, True, True, False, "linear", "reflect", True, {}),
                [[NODATA_OUT]],
                [[0]],
            ),
            (
                (18.01, 27.0, True, True, False, "linear", "reflect", True, {}),
                [[567.300000]],
                [[1]],
            ),
            (
                (15.99, 27.0, True, True, False, "linear", "reflect", True, {}),
                [[506.700000]],
                [[1]],
            ),
            (
                (16.01, 27.0, True, True, False, "linear", "reflect", True, {}),
                [[NODATA_OUT]],
                [[0]],
            ),
            (
                (17.0, 25.99, True, True, False, "linear", "reflect", True, {}),
                [[535.990000]],
                [[1]],
            ),
            (
                (17.0, 26.01, True, True, False, "linear", "reflect", True, {}),
                [[NODATA_OUT]],
                [[0]],
            ),
            (
                (17.0, 27.99, True, True, False, "linear", "reflect", True, {}),
                [[NODATA_OUT]],
                [[0]],
            ),
            (
                (17.0, 28.01, True, True, False, "linear", "reflect", True, {}),
                [[538.010000]],
                [[1]],
            ),
            # ------ cubic ------
            # -------- target invalid point --------
            ((0.0, 0.0, True, True, False, "cubic", "reflect", True, {}), [[NODATA_OUT]], [[0]]),
            ((17.0, 27.0, True, True, False, "cubic", "reflect", True, {}), [[NODATA_OUT]], [[0]]),
            # -------- target neighbor point - weight for invalid data should be 0 --------
            ((19.0, 27.0, True, True, False, "cubic", "reflect", True, {}), [[597.0]], [[1]]),
            ((15.0, 27.0, True, True, False, "cubic", "reflect", True, {}), [[477.0]], [[1]]),
            ((17.0, 25.0, True, True, False, "cubic", "reflect", True, {}), [[535.0]], [[1]]),
            ((17.0, 29.0, True, True, False, "cubic", "reflect", True, {}), [[539.0]], [[1]]),
            # -------- target neighbor point - interpolation required --------
            ((14.99, 27.0, True, True, False, "cubic", "reflect", True, {}), [[476.700000]], [[1]]),
            ((15.01, 27.0, True, True, False, "cubic", "reflect", True, {}), [[NODATA_OUT]], [[0]]),
            ((18.99, 27.0, True, True, False, "cubic", "reflect", True, {}), [[NODATA_OUT]], [[0]]),
            # 19.01 is outside of source domain => nodata
            ((19.01, 27.0, True, True, False, "cubic", "reflect", True, {}), [[NODATA_OUT]], [[0]]),
            ((17.0, 24.99, True, True, False, "cubic", "reflect", True, {}), [[534.990000]], [[1]]),
            ((17.0, 25.01, True, True, False, "cubic", "reflect", True, {}), [[NODATA_OUT]], [[0]]),
            ((17.0, 28.99, True, True, False, "cubic", "reflect", True, {}), [[NODATA_OUT]], [[0]]),
            # 29.01 is outside of source domain => nodata
            ((17.0, 29.01, True, True, False, "cubic", "reflect", True, {}), [[NODATA_OUT]], [[0]]),
            # ------ bspline3 ------
            # -------- target invalid point --------
            (
                (0.0, 0.0, True, True, False, "bspline3", "reflect", True, DFLT_BSPLINES_ARGS),
                [[NODATA_OUT]],
                [[0]],
            ),
            (
                (17.0, 27.0, True, True, False, "bspline3", "reflect", True, DFLT_BSPLINES_ARGS),
                [[NODATA_OUT]],
                [[0]],
            ),
            # -------- target neighbor point - weight for invalid data should be 0 --------
            (
                (19.0, 27.0, True, True, False, "bspline3", "reflect", True, DFLT_BSPLINES_ARGS),
                [[597.0]],
                [[1]],
            ),
            (
                (15.0, 27.0, True, True, False, "bspline3", "reflect", True, DFLT_BSPLINES_ARGS),
                [[477.0]],
                [[1]],
            ),
            (
                (17.0, 25.0, True, True, False, "bspline3", "reflect", True, DFLT_BSPLINES_ARGS),
                [[535.0]],
                [[1]],
            ),
            (
                (17.0, 29.0, True, True, False, "bspline3", "reflect", True, DFLT_BSPLINES_ARGS),
                [[539.0]],
                [[1]],
            ),
            # activating mask dilatation during prefiltering results in invalid data on stencil
            (
                (
                    19.0,
                    27.0,
                    True,
                    True,
                    False,
                    "bspline3",
                    "reflect",
                    True,
                    PREFIL_MSK_DILATE_BSPLINES_ARGS,
                ),
                [[NODATA_OUT]],
                [[0]],
            ),
            (
                (
                    15.0,
                    27.0,
                    True,
                    True,
                    False,
                    "bspline3",
                    "reflect",
                    True,
                    PREFIL_MSK_DILATE_BSPLINES_ARGS,
                ),
                [[NODATA_OUT]],
                [[0]],
            ),
            (
                (
                    17.0,
                    25.0,
                    True,
                    True,
                    False,
                    "bspline3",
                    "reflect",
                    True,
                    PREFIL_MSK_DILATE_BSPLINES_ARGS,
                ),
                [[NODATA_OUT]],
                [[0]],
            ),
            (
                (
                    17.0,
                    29.0,
                    True,
                    True,
                    False,
                    "bspline3",
                    "reflect",
                    True,
                    PREFIL_MSK_DILATE_BSPLINES_ARGS,
                ),
                [[NODATA_OUT]],
                [[0]],
            ),
            # -------- target neighbor point - interpolation required --------
            (
                (14.99, 27.0, True, True, False, "bspline3", "reflect", True, DFLT_BSPLINES_ARGS),
                [[476.701221]],
                [[1]],
            ),
            (
                (15.01, 27.0, True, True, False, "bspline3", "reflect", True, DFLT_BSPLINES_ARGS),
                [[NODATA_OUT]],
                [[0]],
            ),
            (
                (18.99, 27.0, True, True, False, "bspline3", "reflect", True, DFLT_BSPLINES_ARGS),
                [[NODATA_OUT]],
                [[0]],
            ),
            # 19.01 is outside of source domain => nodata
            (
                (19.01, 27.0, True, True, False, "bspline3", "reflect", True, DFLT_BSPLINES_ARGS),
                [[NODATA_OUT]],
                [[0]],
            ),
            (
                (17.0, 24.99, True, True, False, "bspline3", "reflect", True, DFLT_BSPLINES_ARGS),
                [[534.98972]],
                [[1]],
            ),
            (
                (17.0, 25.01, True, True, False, "bspline3", "reflect", True, DFLT_BSPLINES_ARGS),
                [[NODATA_OUT]],
                [[0]],
            ),
            (
                (17.0, 28.99, True, True, False, "bspline3", "reflect", True, DFLT_BSPLINES_ARGS),
                [[NODATA_OUT]],
                [[0]],
            ),
            # 29.01 is outside of source domain => nodata
            (
                (17.0, 29.01, True, True, False, "bspline3", "reflect", True, DFLT_BSPLINES_ARGS),
                [[NODATA_OUT]],
                [[0]],
            ),
        ],
    )
    def test_mono_point_grid__nominal_case(
        self, args, expected_resampled_array, expected_resampled_mask
    ):
        testing_decimal = 6
        """Test Mono Point Grid for Nominal Cases"""
        kwargs = self._mono_point_grid_generate_input__partial_nominal_case(
            **self._mono_point_grid_generate_input__partial_nominal_case_build_kwargs(args)
        )
        kwargs["array_out"] = None  # Force array out allocation
        array_out, mask_out = array_grid_resampling(**kwargs)

        np.testing.assert_array_almost_equal(
            np.atleast_2d(array_out),
            expected_resampled_array,
            decimal=testing_decimal,
        )
        optional_array_equal(
            np.atleast_2d(mask_out),
            expected_resampled_mask,
        )

    @pytest.mark.parametrize("args", [(0, 0), (2, 1), (1, 2), (2, 2)])
    def test_mono_point_grid__invalid_resolution(self, args):
        """Using no (1, 1) resolution with a mono point grid must raise an Exception"""
        # Fake resolution to be (1, 1) for data initialization in order to avoid divide by zero
        kwargs = self._mono_point_grid_generate_input(
            row_target=1.0,
            col_target=1.0,
            array_in_cst_value=1,
            shape_array_in=(20, 30),
            resolution=args,
            use_array_in_mask=False,
            set_array_in_mask_invalid=False,
            set_array_in_mask_safe_window=False,
            array_out_mask=True,
            use_standalone=True,
            interp="linear",
            nodata_out=self.NODATA_OUT,
            array_in_origin=(0.0, 0.0),
            win=None,
            use_grid_mask=False,
            grid_mask_valid_value=0,
            check_boundaries=True,
            boundary_condition=None,
            trust_padding=False,
            bsplines_kwargs=self.DFLT_BSPLINES_ARGS,
        )
        kwargs["array_out"] = None  # Force array out allocation
        with pytest.raises(ValueError) as exc_info:
            _ = array_grid_resampling(**kwargs)
        if args[0] == 0 or args[1] == 0:
            assert "Resolution must be non zero" in str(exc_info.value)
        else:
            assert "InsufficientGridCoverage" in str(exc_info.value)


class TestGridResamplingMultiPointGrid:

    NODATA_OUT = -9999.0

    # -- helper function for general grid case --
    def _create_grid(
        self, nrow, ncol, origin_pos, origin_node, v_row_y, v_row_x, v_col_y, v_col_x, grid_dtype
    ):
        """Create a grid"""
        x = np.arange(0, ncol, dtype=grid_dtype)
        y = np.arange(0, nrow, dtype=grid_dtype)
        xx, yy = np.meshgrid(x, y)

        xx -= origin_pos[0]
        yy -= origin_pos[1]

        yyy = origin_node[0] + yy * v_row_y + xx * v_col_y
        xxx = origin_node[1] + yy * v_row_x + xx * v_col_x

        return yyy, xxx

    def _multi_points_grid_generate_input(
        self,
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
        bsplines_kwargs=None,
    ):
        """
        interp=interp,
        array_in=image,
        grid_row=np.asarray(grid_row),
        grid_col=np.asarray(grid_col),
        grid_resolution=resolution,
        nodata_out=nodata_out,
        array_in_origin=array_in_origin,
        win=win,
        array_in_mask=array_in_mask,
        array_in_mask_safe_win=array_in_mask_safe_win,
        grid_mask=grid_mask,
        grid_mask_valid_value=grid_mask_valid_value,
        grid_nodata=grid_nodata,
        array_out_mask=True,
        check_boundaries=check_boundaries,
        standalone=standalone,
        boundary_condition=boundary_condition,
        trust_padding=trust_padding,
        """
        if len(shape_array_in) == 3:
            _, array_in_row, array_in_col = shape_array_in
        else:
            array_in_row, array_in_col = shape_array_in
        kwargs = {}
        # -- interp --
        if "bspline" in interp:
            if bsplines_kwargs is None:
                bsplines_kwargs = {"epsilon": 1e-2, "mask_influence_threshold": 1}
            interp = get_interpolator(interp, **bsplines_kwargs)
        else:
            interp = get_interpolator(interp)
        interp.initialize()
        kwargs["interp"] = interp
        # -- array_in --
        if array_in_cst_value is not None:
            kwargs["array_in"] = (
                np.ones(np.prod(shape_array_in), dtype=np.float64).reshape(shape_array_in)
                * array_in_cst_value
            )
        else:
            kwargs["array_in"] = np.arange(np.prod(shape_array_in), dtype=np.float64).reshape(
                shape_array_in
            )
        # -- array_in_origin --
        kwargs["array_in_origin"] = array_in_origin
        # -- resolution --
        kwargs["grid_resolution"] = resolution
        # -- array_in_mask
        kwargs["array_in_mask"] = None
        kwargs["array_in_mask_safe_win"] = None
        if use_array_in_mask:
            array_in_mask = np.ones(kwargs["array_in"].shape, dtype=np.uint8, order="C")
            if array_in_mask_invalid_slice is not None:
                array_in_mask[array_in_mask_invalid_slice] = 0
            kwargs["array_in_mask"] = array_in_mask
            if array_in_mask_safe_window:
                kwargs["array_in_mask_safe_win"] = np.asarray(array_in_mask_safe_window)

        grid_row, grid_col = create_grid(**create_grid_kwargs)
        kwargs["grid_row"] = grid_row
        kwargs["grid_col"] = grid_col

        kwargs["array_out_mask"] = array_out_mask

        grid_mask = None
        if use_grid_mask:
            grid_mask = np.full(np.asarray(grid_row).shape, grid_mask_valid_value, dtype=np.uint8)
            if grid_mask_novalid_slice is not None:
                grid_mask[grid_mask_novalid_slice] = grid_mask_valid_value - 1
        kwargs["grid_mask"] = grid_mask
        kwargs["grid_mask_valid_value"] = grid_mask_valid_value
        kwargs["grid_nodata"] = None

        kwargs["nodata_out"] = nodata_out
        kwargs["win"] = win
        if kwargs["win"] is not None:
            kwargs["win"] = np.asarray(kwargs["win"])

        kwargs["check_boundaries"] = check_boundaries
        kwargs["standalone"] = use_standalone
        kwargs["boundary_condition"] = boundary_condition
        kwargs["trust_padding"] = trust_padding

        return kwargs

    # -- Helper for simple general cases without array mask--
    _multi_point_grid_generate_input__partial_nominal_case_no_array_mask = partialmethod(
        _multi_points_grid_generate_input,
        shape_array_in=(10, 15),
        # resolution = (1, 1),
        use_array_in_mask=False,
        array_in_mask_invalid_slice=None,
        array_in_mask_safe_window=None,
        array_out_mask=True,
        use_standalone=True,
        nodata_out=-9999.0,
        array_in_origin=(0.0, 0.0),
        win=None,
        check_boundaries=True,
    )

    def _multi_point_grid_generate_input__partial_nominal_case_no_array_mask_build_kwargs(
        self, args
    ):
        return dict(
            zip(
                (
                    "create_grid_kwargs",
                    "resolution",
                    "array_in_cst_value",
                    "use_grid_mask",
                    "grid_mask_valid_value",
                    "grid_mask_novalid_slice",
                    "boundary_condition",
                    "trust_padding",
                    "interp",
                    "bsplines_kwargs",
                ),
                args,
                strict=True,
            )
        )

    IDENDITY_GRID_2x3_KWARGS = {
        "nrow": 2,
        "ncol": 3,
        "origin_pos": (0, 0),
        "origin_node": (1.0, 1.0),
        "v_row_y": 2.0,
        "v_row_x": 0.0,
        "v_col_y": 0.0,
        "v_col_x": 3.0,
        "grid_dtype": np.float64,
    }

    @pytest.mark.parametrize(
        "interp, interp_kwargs",
        [
            (
                "nearest",
                {},
            ),
            (
                "linear",
                {},
            ),
            (
                "cubic",
                {},
            ),
            # Force good enough precision for epsilon to check for uniformity preservation
            (
                "bspline3",
                {"epsilon": 1e-6, "mask_influence_threshold": 1},
            ),
        ],
    )
    @pytest.mark.parametrize(
        "args, expected_resampled_array, expected_resampled_mask",
        [
            # -- Tests idendity--
            (
                (
                    IDENDITY_GRID_2x3_KWARGS,
                    (1, 1),
                    100.0,
                    False,
                    1,
                    None,
                    "reflect",
                    True,
                ),
                [[100.000000, 100.000000, 100.000000], [100.000000, 100.000000, 100.000000]],
                [[1, 1, 1], [1, 1, 1]],
            ),
            (
                # -- Test resolution on uniform array => this also checks uniformity preservation --
                (IDENDITY_GRID_2x3_KWARGS, (2, 3), 100.0, False, 1, None, "reflect", True),
                [
                    [
                        100.000000,
                        100.000000,
                        100.000000,
                        100.000000,
                        100.000000,
                        100.000000,
                        100.000000,
                    ],
                    [
                        100.000000,
                        100.000000,
                        100.000000,
                        100.000000,
                        100.000000,
                        100.000000,
                        100.000000,
                    ],
                    [
                        100.000000,
                        100.000000,
                        100.000000,
                        100.000000,
                        100.000000,
                        100.000000,
                        100.000000,
                    ],
                ],
                [[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]],
            ),
            (
                # -- Test resolution on uniform array => this also checks uniformity preservation --
                (IDENDITY_GRID_2x3_KWARGS, (200, 300), 100.0, False, 1, None, "reflect", True),
                np.full((201, 601), 100.0),
                np.ones((201, 601)),
            ),
        ],
    )
    def test_multi_point_grid__nominal_case_no_array_mask_uniform_data(
        self, interp, interp_kwargs, args, expected_resampled_array, expected_resampled_mask
    ):
        testing_decimal = 6
        """Test Mono Point Grid for Nominal Cases"""
        args = args + (interp, interp_kwargs)
        kwargs = self._multi_point_grid_generate_input__partial_nominal_case_no_array_mask(
            **self._multi_point_grid_generate_input__partial_nominal_case_no_array_mask_build_kwargs(  # noqa: E501
                args
            )
        )
        kwargs["array_out"] = None  # Force array out allocation
        array_out, mask_out = array_grid_resampling(**kwargs)

        np.testing.assert_array_almost_equal(
            np.atleast_2d(array_out),
            expected_resampled_array,
            decimal=testing_decimal,
        )
        optional_array_equal(
            np.atleast_2d(mask_out),
            expected_resampled_mask,
        )

    GRID_2x3_SHIFT_ROW_KWARGS = copy.deepcopy(IDENDITY_GRID_2x3_KWARGS)
    GRID_2x3_SHIFT_ROW_KWARGS["origin_node"] = (1.5, 1.0)
    GRID_2x3_SHIFT_COL_KWARGS = copy.deepcopy(IDENDITY_GRID_2x3_KWARGS)
    GRID_2x3_SHIFT_COL_KWARGS["origin_node"] = (1.0, 1.2)

    @pytest.mark.parametrize(
        "args, expected_resampled_array, expected_resampled_mask",
        [
            (
                (
                    GRID_2x3_SHIFT_ROW_KWARGS,
                    (1, 1),
                    None,
                    False,
                    1,
                    None,
                    "reflect",
                    True,
                    "linear",
                    {},
                ),
                [[23.500000, 26.500000, 29.500000], [53.500000, 56.500000, 59.500000]],
                [[1, 1, 1], [1, 1, 1]],
            ),
            (
                (
                    GRID_2x3_SHIFT_ROW_KWARGS,
                    (3, 2),
                    None,
                    False,
                    1,
                    None,
                    "reflect",
                    True,
                    "linear",
                    {},
                ),
                [
                    [23.500000, 25.000000, 26.500000, 28.000000, 29.500000],
                    [33.500000, 35.000000, 36.500000, 38.000000, 39.500000],
                    [43.500000, 45.000000, 46.500000, 48.000000, 49.500000],
                    [53.500000, 55.000000, 56.500000, 58.000000, 59.500000],
                ],
                [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
            ),
            (
                (
                    GRID_2x3_SHIFT_COL_KWARGS,
                    (1, 1),
                    None,
                    False,
                    1,
                    None,
                    "reflect",
                    True,
                    "linear",
                    {},
                ),
                [[16.200000, 19.200000, 22.200000], [46.200000, 49.200000, 52.200000]],
                [[1, 1, 1], [1, 1, 1]],
            ),
            (
                (
                    GRID_2x3_SHIFT_COL_KWARGS,
                    (3, 2),
                    None,
                    False,
                    1,
                    None,
                    "reflect",
                    True,
                    "linear",
                    {},
                ),
                [
                    [16.200000, 17.700000, 19.200000, 20.700000, 22.200000],
                    [26.200000, 27.700000, 29.200000, 30.700000, 32.200000],
                    [36.200000, 37.700000, 39.200000, 40.700000, 42.200000],
                    [46.200000, 47.700000, 49.200000, 50.700000, 52.200000],
                ],
                [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
            ),
        ],
    )
    def test_multi_point_grid__nominal_case_no_array_mask_shift(
        self, args, expected_resampled_array, expected_resampled_mask
    ):
        testing_decimal = 6
        """Test Multi Point Grid for Nominal Cases - shift"""
        kwargs = self._multi_point_grid_generate_input__partial_nominal_case_no_array_mask(
            **self._multi_point_grid_generate_input__partial_nominal_case_no_array_mask_build_kwargs(  # noqa: E501
                args
            )
        )
        kwargs["array_out"] = None  # Force array out allocation
        array_out, mask_out = array_grid_resampling(**kwargs)

        np.testing.assert_array_almost_equal(
            np.atleast_2d(array_out),
            expected_resampled_array,
            decimal=testing_decimal,
        )
        optional_array_equal(
            np.atleast_2d(mask_out),
            expected_resampled_mask,
        )

    # -- Helper for grid mask and windowing --
    _multi_point_grid_generate_input__partial_grid_mask_and_win = partialmethod(
        _multi_points_grid_generate_input,
        shape_array_in=(10, 15),
        array_in_cst_value=None,
        use_array_in_mask=False,
        array_in_mask_invalid_slice=None,
        array_in_mask_safe_window=None,
        array_out_mask=True,
        use_standalone=True,
        nodata_out=-9999.0,
        array_in_origin=(0.0, 0.0),
        check_boundaries=True,
    )

    def _multi_point_grid_generate_input__partial_grid_mask_and_win_build_kwargs(self, args):
        return dict(
            zip(
                (
                    "create_grid_kwargs",
                    "resolution",
                    "win",
                    "use_grid_mask",
                    "grid_mask_valid_value",
                    "grid_mask_novalid_slice",
                    "boundary_condition",
                    "trust_padding",
                    "interp",
                    "bsplines_kwargs",
                ),
                args,
                strict=True,
            )
        )

    IDENDITY_GRID_3x3_KWARGS = {
        "nrow": 3,
        "ncol": 3,
        "origin_pos": (0, 0),
        "origin_node": (1.0, 1.2),
        "v_row_y": 2.0,
        "v_row_x": 0.0,
        "v_col_y": 0.0,
        "v_col_x": 3.0,
        "grid_dtype": np.float64,
    }

    # also check for windowing
    @pytest.mark.parametrize(
        "win",
        [
            None,
            ((0, 2), (0, 2)),
            ((1, 1), (1, 1)),
            ((0, 0), (0, 0)),
            ((0, 0), (2, 2)),
            ((2, 2), (0, 0)),
            ((2, 2), (2, 2)),
        ],
    )
    @pytest.mark.parametrize(
        "args, expected_resampled_array, expected_resampled_mask",
        [
            # Full invalid grid mask - value = 1
            (
                (
                    IDENDITY_GRID_3x3_KWARGS,
                    (1, 1),
                    None,
                    True,
                    1,
                    (slice(None), slice(None)),
                    "reflect",
                    True,
                    "linear",
                    {},
                ),
                np.full((3, 3), NODATA_OUT),
                np.full((3, 3), 0),
            ),
            # Full invalid grid mask - value = 2
            (
                (
                    IDENDITY_GRID_3x3_KWARGS,
                    (1, 1),
                    None,
                    True,
                    2,
                    (slice(None), slice(None)),
                    "reflect",
                    True,
                    "linear",
                    {},
                ),
                np.full((3, 3), NODATA_OUT),
                np.full((3, 3), 0),
            ),
            # Full valid grid mask
            (
                (
                    IDENDITY_GRID_3x3_KWARGS,
                    (1, 1),
                    None,
                    True,
                    1,
                    None,
                    "reflect",
                    True,
                    "linear",
                    {},
                ),
                [
                    [16.200000, 19.200000, 22.200000],
                    [46.200000, 49.200000, 52.200000],
                    [76.200000, 79.200000, 82.200000],
                ],
                np.full((3, 3), 1),
            ),
            # Upper Left - invalid grid mask - res = (1, 1)
            (
                (
                    IDENDITY_GRID_3x3_KWARGS,
                    (1, 1),
                    None,
                    True,
                    1,
                    (slice(0, 1), slice(0, 1)),
                    "reflect",
                    True,
                    "linear",
                    {},
                ),
                [
                    [NODATA_OUT, 19.200000, 22.200000],
                    [46.200000, 49.200000, 52.200000],
                    [76.200000, 79.200000, 82.200000],
                ],
                [[0, 1, 1], [1, 1, 1], [1, 1, 1]],
            ),
            # Bottom Left - invalid grid mask - res = (1, 1)
            (
                (
                    IDENDITY_GRID_3x3_KWARGS,
                    (1, 1),
                    None,
                    True,
                    1,
                    (slice(2, 3), slice(0, 1)),
                    "reflect",
                    True,
                    "linear",
                    {},
                ),
                [
                    [16.200000, 19.200000, 22.200000],
                    [46.200000, 49.200000, 52.200000],
                    [NODATA_OUT, 79.200000, 82.200000],
                ],
                [[1, 1, 1], [1, 1, 1], [0, 1, 1]],
            ),
            # Upper Right - invalid grid mask - res = (1, 1)
            (
                (
                    IDENDITY_GRID_3x3_KWARGS,
                    (1, 1),
                    None,
                    True,
                    1,
                    (slice(0, 1), slice(2, 3)),
                    "reflect",
                    True,
                    "linear",
                    {},
                ),
                [
                    [16.200000, 19.200000, NODATA_OUT],
                    [46.200000, 49.200000, 52.200000],
                    [76.200000, 79.200000, 82.200000],
                ],
                [[1, 1, 0], [1, 1, 1], [1, 1, 1]],
            ),
            # Bottom Right - invalid grid mask - res = (1, 1)
            (
                (
                    IDENDITY_GRID_3x3_KWARGS,
                    (1, 1),
                    None,
                    True,
                    1,
                    (slice(2, 3), slice(2, 3)),
                    "reflect",
                    True,
                    "linear",
                    {},
                ),
                [
                    [16.200000, 19.200000, 22.200000],
                    [46.200000, 49.200000, 52.200000],
                    [76.200000, 79.200000, NODATA_OUT],
                ],
                [[1, 1, 1], [1, 1, 1], [1, 1, 0]],
            ),
            # Center - invalid grid mask - res = (1, 1)
            (
                (
                    IDENDITY_GRID_3x3_KWARGS,
                    (1, 1),
                    None,
                    True,
                    1,
                    (slice(1, 2), slice(1, 2)),
                    "reflect",
                    True,
                    "linear",
                    {},
                ),
                [
                    [16.200000, 19.200000, 22.200000],
                    [46.200000, NODATA_OUT, 52.200000],
                    [76.200000, 79.200000, 82.200000],
                ],
                [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
            ),
        ],
    )
    def test_multi_point_grid__grid_mask_and_out_window__resolution_1_1(
        self, win, args, expected_resampled_array, expected_resampled_mask
    ):
        testing_decimal = 6

        # Run with defined window in args - must be None here
        assert args[2] is None
        """Test Multi Point Grid for Nominal Cases - shift"""
        kwargs = self._multi_point_grid_generate_input__partial_grid_mask_and_win(
            **self._multi_point_grid_generate_input__partial_grid_mask_and_win_build_kwargs(args)
        )
        kwargs["array_out"] = None  # Force array out allocation
        array_out, mask_out = array_grid_resampling(**kwargs)

        np.testing.assert_array_almost_equal(
            np.atleast_2d(array_out),
            expected_resampled_array,
            decimal=testing_decimal,
        )
        optional_array_equal(
            np.atleast_2d(mask_out),
            expected_resampled_mask,
        )

        # Run with param win
        # overwrite window args component
        if win is not None:
            args = args[:2] + (win,) + args[3:]
            kwargs = self._multi_point_grid_generate_input__partial_grid_mask_and_win(
                **self._multi_point_grid_generate_input__partial_grid_mask_and_win_build_kwargs(
                    args
                )
            )
            kwargs["array_out"] = None  # Force array out allocation
            array_out_win, mask_out_win = array_grid_resampling(**kwargs)

            indices = (..., *window_indices(win))
            np.testing.assert_array_almost_equal(
                np.atleast_2d(array_out_win),
                np.atleast_2d(array_out)[indices],
                decimal=testing_decimal,
            )
            optional_array_equal(
                np.atleast_2d(mask_out_win),
                np.atleast_2d(mask_out)[indices],
            )

    # also check for windowing
    @pytest.mark.parametrize(
        "win",
        [
            None,
            ((0, 4), (0, 6)),
            ((1, 3), (1, 5)),
            ((0, 0), (0, 0)),
            ((0, 0), (6, 6)),
            ((4, 4), (0, 0)),
            ((4, 4), (6, 6)),
            ((2, 2), (2, 2)),
        ],
    )
    @pytest.mark.parametrize(
        "args, expected_resampled_array, expected_resampled_mask",
        [
            # Full invalid grid mask - value = 1
            (
                (
                    IDENDITY_GRID_3x3_KWARGS,
                    (2, 3),
                    None,
                    True,
                    1,
                    (slice(None), slice(None)),
                    "reflect",
                    True,
                    "linear",
                    {},
                ),
                np.full((5, 7), NODATA_OUT),
                np.full((5, 7), 0),
            ),
            # Full valid grid mask
            (
                (
                    IDENDITY_GRID_3x3_KWARGS,
                    (2, 3),
                    None,
                    True,
                    1,
                    None,
                    "reflect",
                    True,
                    "linear",
                    {},
                ),
                [
                    [16.200000, 17.200000, 18.200000, 19.200000, 20.200000, 21.200000, 22.200000],
                    [31.200000, 32.200000, 33.200000, 34.200000, 35.200000, 36.200000, 37.200000],
                    [46.200000, 47.200000, 48.200000, 49.200000, 50.200000, 51.200000, 52.200000],
                    [61.200000, 62.200000, 63.200000, 64.200000, 65.200000, 66.200000, 67.200000],
                    [76.200000, 77.200000, 78.200000, 79.200000, 80.200000, 81.200000, 82.200000],
                ],
                np.full((5, 7), 1),
            ),
            # Upper Left - invalid grid mask - res = (2, 3)
            (
                (
                    IDENDITY_GRID_3x3_KWARGS,
                    (2, 3),
                    None,
                    True,
                    1,
                    (slice(0, 1), slice(0, 1)),
                    "reflect",
                    True,
                    "linear",
                    {},
                ),
                [
                    [
                        NODATA_OUT,
                        NODATA_OUT,
                        NODATA_OUT,
                        19.200000,
                        20.200000,
                        21.200000,
                        22.200000,
                    ],
                    [
                        NODATA_OUT,
                        NODATA_OUT,
                        NODATA_OUT,
                        34.200000,
                        35.200000,
                        36.200000,
                        37.200000,
                    ],
                    [46.200000, 47.200000, 48.200000, 49.200000, 50.200000, 51.200000, 52.200000],
                    [61.200000, 62.200000, 63.200000, 64.200000, 65.200000, 66.200000, 67.200000],
                    [76.200000, 77.200000, 78.200000, 79.200000, 80.200000, 81.200000, 82.200000],
                ],
                [
                    [0, 0, 0, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                ],
            ),
            # Bottom Left - invalid grid mask - res = (2, 3)
            (
                (
                    IDENDITY_GRID_3x3_KWARGS,
                    (2, 3),
                    None,
                    True,
                    1,
                    (slice(2, 3), slice(0, 1)),
                    "reflect",
                    True,
                    "linear",
                    {},
                ),
                [
                    [16.200000, 17.200000, 18.200000, 19.200000, 20.200000, 21.200000, 22.200000],
                    [31.200000, 32.200000, 33.200000, 34.200000, 35.200000, 36.200000, 37.200000],
                    [46.200000, 47.200000, 48.200000, 49.200000, 50.200000, 51.200000, 52.200000],
                    [
                        NODATA_OUT,
                        NODATA_OUT,
                        NODATA_OUT,
                        64.200000,
                        65.200000,
                        66.200000,
                        67.200000,
                    ],
                    [
                        NODATA_OUT,
                        NODATA_OUT,
                        NODATA_OUT,
                        79.200000,
                        80.200000,
                        81.200000,
                        82.200000,
                    ],
                ],
                [
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1],
                ],
            ),
            # Upper Right - invalid grid mask - res = (2, 3)
            (
                (
                    IDENDITY_GRID_3x3_KWARGS,
                    (2, 3),
                    None,
                    True,
                    1,
                    (slice(0, 1), slice(2, 3)),
                    "reflect",
                    True,
                    "linear",
                    {},
                ),
                [
                    [
                        16.200000,
                        17.200000,
                        18.200000,
                        19.200000,
                        NODATA_OUT,
                        NODATA_OUT,
                        NODATA_OUT,
                    ],
                    [
                        31.200000,
                        32.200000,
                        33.200000,
                        34.200000,
                        NODATA_OUT,
                        NODATA_OUT,
                        NODATA_OUT,
                    ],
                    [46.200000, 47.200000, 48.200000, 49.200000, 50.200000, 51.200000, 52.200000],
                    [61.200000, 62.200000, 63.200000, 64.200000, 65.200000, 66.200000, 67.200000],
                    [76.200000, 77.200000, 78.200000, 79.200000, 80.200000, 81.200000, 82.200000],
                ],
                [
                    [1, 1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                ],
            ),
            # Bottom Right - invalid grid mask - res = (2, 3)
            (
                (
                    IDENDITY_GRID_3x3_KWARGS,
                    (2, 3),
                    None,
                    True,
                    1,
                    (slice(2, 3), slice(2, 3)),
                    "reflect",
                    True,
                    "linear",
                    {},
                ),
                [
                    [16.200000, 17.200000, 18.200000, 19.200000, 20.200000, 21.200000, 22.200000],
                    [31.200000, 32.200000, 33.200000, 34.200000, 35.200000, 36.200000, 37.200000],
                    [46.200000, 47.200000, 48.200000, 49.200000, 50.200000, 51.200000, 52.200000],
                    [
                        61.200000,
                        62.200000,
                        63.200000,
                        64.200000,
                        NODATA_OUT,
                        NODATA_OUT,
                        NODATA_OUT,
                    ],
                    [
                        76.200000,
                        77.200000,
                        78.200000,
                        79.200000,
                        NODATA_OUT,
                        NODATA_OUT,
                        NODATA_OUT,
                    ],
                ],
                [
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0],
                ],
            ),
            # Center - invalid grid mask - res = (2, 3)
            (
                (
                    IDENDITY_GRID_3x3_KWARGS,
                    (2, 3),
                    None,
                    True,
                    1,
                    (slice(1, 2), slice(1, 2)),
                    "reflect",
                    True,
                    "linear",
                    {},
                ),
                [
                    [16.200000, 17.200000, 18.200000, 19.200000, 20.200000, 21.200000, 22.200000],
                    [
                        31.200000,
                        NODATA_OUT,
                        NODATA_OUT,
                        NODATA_OUT,
                        NODATA_OUT,
                        NODATA_OUT,
                        37.200000,
                    ],
                    [
                        46.200000,
                        NODATA_OUT,
                        NODATA_OUT,
                        NODATA_OUT,
                        NODATA_OUT,
                        NODATA_OUT,
                        52.200000,
                    ],
                    [
                        61.200000,
                        NODATA_OUT,
                        NODATA_OUT,
                        NODATA_OUT,
                        NODATA_OUT,
                        NODATA_OUT,
                        67.200000,
                    ],
                    [76.200000, 77.200000, 78.200000, 79.200000, 80.200000, 81.200000, 82.200000],
                ],
                [
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                ],
            ),
        ],
    )
    def test_multi_point_grid__grid_mask_and_out_window__resolution_2_3(
        self, win, args, expected_resampled_array, expected_resampled_mask
    ):
        testing_decimal = 6
        # Run with defined window in args - must be None here
        assert args[2] is None
        """Test Multi Point Grid for Nominal Cases - shift"""
        kwargs = self._multi_point_grid_generate_input__partial_grid_mask_and_win(
            **self._multi_point_grid_generate_input__partial_grid_mask_and_win_build_kwargs(args)
        )
        kwargs["array_out"] = None  # Force array out allocation
        array_out, mask_out = array_grid_resampling(**kwargs)

        np.testing.assert_array_almost_equal(
            np.atleast_2d(array_out),
            expected_resampled_array,
            decimal=testing_decimal,
        )
        optional_array_equal(
            np.atleast_2d(mask_out),
            expected_resampled_mask,
        )

        # Run with param win
        # overwrite window args component
        if win is not None:
            args = args[:2] + (win,) + args[3:]
            kwargs = self._multi_point_grid_generate_input__partial_grid_mask_and_win(
                **self._multi_point_grid_generate_input__partial_grid_mask_and_win_build_kwargs(
                    args
                )
            )
            kwargs["array_out"] = None  # Force array out allocation
            array_out_win, mask_out_win = array_grid_resampling(**kwargs)

            indices = (..., *window_indices(win))
            np.testing.assert_array_almost_equal(
                np.atleast_2d(array_out_win),
                np.atleast_2d(array_out)[indices],
                decimal=testing_decimal,
            )
            optional_array_equal(
                np.atleast_2d(mask_out_win),
                np.atleast_2d(mask_out)[indices],
            )

    @pytest.mark.parametrize(
        "interp, interp_kwargs",
        [
            ("nearest", None),
            ("linear", None),
            ("cubic", None),
            # ("bspline3", {"epsilon": 1e-6, "mask_influence_threshold": 1}),
        ],
    )
    @pytest.mark.parametrize("boundary_condition", ["reflect", "symmetric", "constant", None])
    @pytest.mark.parametrize("trust_padding", [True, False])
    @pytest.mark.parametrize("use_standalone", [True, False])
    def test_multi_point_grid__safe_window__nominal_case(
        self,
        interp,
        interp_kwargs,
        boundary_condition,
        trust_padding,
        use_standalone,
    ):
        """
        This test checks that using safe window feature provides the same
        results as not using it regardless of the mathematic results (that may
        be incorrect with use_standalone set as `False`
        """
        testing_decimal = 6

        grid_kwargs = {
            "nrow": 10,
            "ncol": 10,
            "origin_pos": (0, 0),
            "origin_node": (19.0, 20.0),
            "v_row_y": 2.30,
            "v_row_x": 0.2,
            "v_col_y": 0.26,
            "v_col_x": 1.6,
            "grid_dtype": np.float64,
        }

        # no safe window
        kwargs_0 = self._multi_points_grid_generate_input(
            array_in_cst_value=None,
            shape_array_in=(50, 60),
            create_grid_kwargs=grid_kwargs,
            resolution=(1, 1),
            use_array_in_mask=True,
            array_in_mask_invalid_slice=(slice(4, 30), slice(13, 23)),
            array_in_mask_safe_window=None,
            array_out_mask=True,
            use_standalone=use_standalone,
            interp=interp,
            nodata_out=self.NODATA_OUT,
            array_in_origin=(0.0, 0.0),
            win=None,
            use_grid_mask=None,
            grid_mask_valid_value=1,
            grid_mask_novalid_slice=None,
            check_boundaries=True,
            boundary_condition=boundary_condition,
            trust_padding=trust_padding,
            bsplines_kwargs=interp_kwargs,
        )
        kwargs_0["array_out"] = None

        # correct safe window
        kwargs_1 = self._multi_points_grid_generate_input(
            array_in_cst_value=None,
            shape_array_in=(50, 60),
            create_grid_kwargs=grid_kwargs,
            resolution=(1, 1),
            use_array_in_mask=True,
            array_in_mask_invalid_slice=(slice(4, 30), slice(13, 23)),
            array_in_mask_safe_window=((30, 49), (23, 59)),
            array_out_mask=True,
            use_standalone=use_standalone,
            interp=interp,
            nodata_out=self.NODATA_OUT,
            array_in_origin=(0.0, 0.0),
            win=None,
            use_grid_mask=None,
            grid_mask_valid_value=1,
            grid_mask_novalid_slice=None,
            check_boundaries=True,
            boundary_condition=boundary_condition,
            trust_padding=trust_padding,
            bsplines_kwargs=interp_kwargs,
        )
        kwargs_1["array_out"] = None

        # bad safe window
        kwargs_2 = self._multi_points_grid_generate_input(
            array_in_cst_value=None,
            shape_array_in=(50, 60),
            create_grid_kwargs=grid_kwargs,
            resolution=(1, 1),
            use_array_in_mask=True,
            array_in_mask_invalid_slice=(slice(4, 30), slice(13, 23)),
            array_in_mask_safe_window=((10, 49), (10, 59)),
            array_out_mask=True,
            use_standalone=use_standalone,
            interp=interp,
            nodata_out=self.NODATA_OUT,
            array_in_origin=(0.0, 0.0),
            win=None,
            use_grid_mask=None,
            grid_mask_valid_value=1,
            grid_mask_novalid_slice=None,
            check_boundaries=True,
            boundary_condition=boundary_condition,
            trust_padding=trust_padding,
            bsplines_kwargs=interp_kwargs,
        )
        kwargs_2["array_out"] = None

        # no safe window
        array_out_0, mask_out_0 = array_grid_resampling(**kwargs_0)
        # correct safe window
        array_out_1, mask_out_1 = array_grid_resampling(**kwargs_1)
        # bad safe window
        array_out_2, mask_out_2 = array_grid_resampling(**kwargs_2)

        # correct safe window must not change results
        np.testing.assert_array_almost_equal(
            np.atleast_2d(array_out_1),
            np.atleast_2d(array_out_0),
            decimal=testing_decimal,
        )
        optional_array_equal(
            np.atleast_2d(mask_out_1),
            np.atleast_2d(mask_out_0),
        )

        # The nearest interpolation core does not use safe window as there is
        # no convolution during interpolation
        if interp not in [
            "nearest",
        ]:
            # bad safe window must change results
            with pytest.raises(AssertionError):
                np.testing.assert_array_almost_equal(
                    np.atleast_2d(array_out_2),
                    np.atleast_2d(array_out_0),
                    decimal=testing_decimal,
                )
            if optional_array_equal(
                np.atleast_2d(mask_out_2),
                np.atleast_2d(mask_out_0),
            ):
                raise AssertionError("Bad safe window case : masks must be different !")
        else:
            np.testing.assert_array_almost_equal(
                np.atleast_2d(array_out_2),
                np.atleast_2d(array_out_0),
                decimal=testing_decimal,
            )
            optional_array_equal(
                np.atleast_2d(mask_out_2),
                np.atleast_2d(mask_out_0),
            )
