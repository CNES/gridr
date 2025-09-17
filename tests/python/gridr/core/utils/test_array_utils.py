# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of GRIDR
# (see https://gitlab.cnes.fr/gridr/gridr).
#
#
"""
Tests for the gridr.core.utils.array_window module

Command to run test :
PYTHONPATH=${PWD}/python/:$PYTHONPATH pytest tests/python/gridr/core/utils/test_array_window.py
"""
import numpy as np
import pytest

from gridr.core.utils.array_utils import array_replace

ARRAY_I8_00 = np.zeros(4 * 7, dtype=np.int8).reshape(4, 7)
ARRAY_I8_00[1, :] = 1

ARRAY_U8_00 = np.zeros(4 * 7, dtype=np.uint8).reshape(4, 7)
ARRAY_U8_00[1, :] = 1

ARRAY_I32_00 = np.zeros(4 * 7, dtype=np.int32).reshape(4, 7)
ARRAY_I32_00[1, :] = 1

ARRAY_F32_00 = np.zeros(4 * 7, dtype=np.float32).reshape(4, 7)
ARRAY_F32_00[1, :] = 1

ARRAY_F64_00 = np.zeros(4 * 7, dtype=np.float64).reshape(4, 7)
ARRAY_F64_00[1, :] = 1

WIN_00 = [(0, 2), (2, 4)]

ARRAY_COND_U8 = np.zeros(4 * 7, dtype=np.uint8).reshape(4, 7)
ARRAY_COND_U8[0:2, 0:3] = 98

ARRAY_F32_00_expected_cond = np.copy(ARRAY_F32_00)
ARRAY_F32_00_val_cond = 98
ARRAY_F32_00_val_true = 999
ARRAY_F32_00_expected_cond[ARRAY_COND_U8 == ARRAY_F32_00_val_cond] = ARRAY_F32_00_val_true

ARRAY_F64_00_expected_cond = np.copy(ARRAY_F64_00)
ARRAY_F64_00_val_cond = 98
ARRAY_F64_00_val_true = 9999
ARRAY_F64_00_expected_cond[ARRAY_COND_U8 == ARRAY_F64_00_val_cond] = ARRAY_F64_00_val_true


class TestArrayWindow:
    """Class for test"""

    @pytest.mark.parametrize(
        "data, expected",
        [
            ((ARRAY_I8_00, 0, 0, 1, None, None), ARRAY_I8_00),
            ((ARRAY_I8_00, 0, 1, 0, None, None), np.where(ARRAY_I8_00 == 0, 1, 0)),
            ((ARRAY_U8_00, 0, 0, 1, None, None), ARRAY_U8_00),
            ((ARRAY_I32_00, 0, 0, 1, None, None), AssertionError),
            (
                (
                    ARRAY_F32_00,
                    0,
                    ARRAY_F32_00_val_true,
                    None,
                    ARRAY_COND_U8,
                    ARRAY_F32_00_val_cond,
                ),
                ARRAY_F32_00_expected_cond,
            ),
            (
                (
                    ARRAY_F64_00,
                    0,
                    ARRAY_F64_00_val_true,
                    None,
                    ARRAY_COND_U8,
                    ARRAY_F64_00_val_cond,
                ),
                ARRAY_F64_00_expected_cond,
            ),
            ((ARRAY_I8_00, 1, 3, 2, None, None), np.where(ARRAY_I8_00 == 1, 3, 2)),
        ],
    )
    @pytest.mark.parametrize("win", [None, WIN_00])
    def test_array_replace(self, data, expected, win):
        """Test array_replace method"""
        array, val_cond, val_true, val_false, array_cond, array_val_cond = data
        expected_array = expected
        window = win
        # Copy input array in order to not affect the global variable
        array_copy = np.copy(array)

        if window is not None:
            expected_array_win = np.copy(array)
            win_slice = (
                slice(WIN_00[0][0], WIN_00[0][1] + 1),
                slice(WIN_00[1][0], WIN_00[1][1] + 1),
            )
            try:
                expected_array_win[win_slice] = expected_array[win_slice]
                expected_array = expected_array_win
            except TypeError:
                pass

        try:
            array_replace(
                array_copy, val_cond, val_true, val_false, array_cond, array_val_cond, window
            )
        except Exception as e:
            if isinstance(e, expected):
                pass
            else:
                raise
        else:
            assert np.all(array_copy == expected_array)

    @pytest.mark.parametrize(
        "data, expected",
        [
            ((ARRAY_I8_00, 0, 1, 0, (slice(0, 4), slice(0, 7))), np.where(ARRAY_I8_00 == 0, 1, 0)),
            # Testing slices : here take all lines but select only some columns => the view is not
            # a contiguous view => an AssertionError must be raised
            ((ARRAY_I8_00, 0, 1, 0, (slice(0, 4), slice(2, 4))), AssertionError),
            # Testing slices : here take somes lines but select all columns => the view is still
            # a contiguous view => it should be OK
            ((ARRAY_I8_00, 0, 1, 0, (slice(1, 3), slice(0, 7))), np.where(ARRAY_I8_00 == 0, 1, 0)),
        ],
    )
    def test_array_replace_contiguous(self, data, expected):
        """Test array_replace method"""
        array, val_cond, val_true, val_false, slices = data
        expected_array = expected

        # Copy input array in order to not affect the global variable
        array_copy = np.copy(array)

        try:
            if slices is None:
                array_replace(array_copy, val_cond, val_true, val_false, None, None, None)
            else:
                array_replace(array_copy[slices], val_cond, val_true, val_false, None, None, None)
        except Exception as e:
            if isinstance(e, expected):
                pass
            else:
                raise
        else:
            if slices is None:
                assert np.all(array_copy == expected_array)
            else:
                assert np.all(array_copy[slices] == expected_array[slices])
