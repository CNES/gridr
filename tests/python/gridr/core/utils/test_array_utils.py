# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of GRIDR
# (see https://gitlab.cnes.fr/gridr/gridr).
#
#
"""
Tests for the gridr.core.utils.array_utils module

Command to run test :
PYTHONPATH=${PWD}/python/:$PYTHONPATH pytest tests/python/gridr/core/utils/test_array_utils.py
"""
import numpy as np
import pytest

from gridr.core.utils.array_utils import (
    array_add,
    array_convert,
    array_replace,
    is_clip_required,
    is_clip_to_dtype_limits_safe,
)

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

ARRAY_F32_00_expected_cond_add_on_true_true = np.copy(ARRAY_F32_00)
ARRAY_F32_00_expected_cond_add_on_true_true[ARRAY_COND_U8 == ARRAY_F32_00_val_cond] += 10

ARRAY_F32_00_expected_cond_add_on_true_false = np.copy(ARRAY_F32_00)
ARRAY_F32_00_expected_cond_add_on_true_false[ARRAY_COND_U8 != ARRAY_F32_00_val_cond] += 10

ARRAY_F64_00_expected_cond = np.copy(ARRAY_F64_00)
ARRAY_F64_00_val_cond = 98
ARRAY_F64_00_val_true = 9999
ARRAY_F64_00_expected_cond[ARRAY_COND_U8 == ARRAY_F64_00_val_cond] = ARRAY_F64_00_val_true

ARRAY_F64_00_expected_cond_add_on_true_true = np.copy(ARRAY_F64_00)
ARRAY_F64_00_expected_cond_add_on_true_true[ARRAY_COND_U8 == ARRAY_F64_00_val_cond] += 10

ARRAY_CONVERT_DATA = {
    "int8": np.array([-128, -1, 0, 1, 127], dtype=np.int8),
    "int16": np.array([-32768, -1, 0, 1, 32767], dtype=np.int16),
    "int32": np.array([-2147483648, -1, 0, 1, 2147483647], dtype=np.int32),
    "int64": np.array([-9223372036854775808, -1, 0, 1, 9223372036854775807], dtype=np.int64),
    "uint8": np.array([0, 0, 0, 1, 255], dtype=np.uint8),
    "uint16": np.array([0, 0, 0, 1, 65535], dtype=np.uint16),
    "uint32": np.array([0, 0, 0, 1, 4294967295], dtype=np.uint32),
    "uint64": np.array([0, 0, 0, 1, 18446744073709551615], dtype=np.uint64),
    "float16": np.array([-65504, -1.0, 0.0, 1.0, 65504], dtype=np.float16),
    "float32": np.array([-3.4028235e38, -1.0, 0.0, 1.0, 3.4028235e38], dtype=np.float32),
    "float64": np.array(
        [-1.7976931348623157e308, -1.0, 0.0, 1.0, 1.7976931348623157e308], dtype=np.float64
    ),
}
ARRAY_CONVERT_DATA_ROUNDING_METHOD = {
    "float64": np.array([-1.0, -0.6, -0.5, -0.4, 0.0, 0.1, 0.5, 0.8], dtype=np.float64),
}


class TestArrayUtils:
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

    @pytest.mark.parametrize(
        "data, expected",
        [
            ((ARRAY_I8_00, 0, 0, True, None, None), ARRAY_I8_00),
            ((ARRAY_I8_00, 0, 0, False, None, None), ARRAY_I8_00),
            ((ARRAY_I8_00, 0, 10, True, None, None), np.where(ARRAY_I8_00 == 0, 10, 1)),
            ((ARRAY_I8_00, 0, 10, False, None, None), np.where(ARRAY_I8_00 != 0, 11, 0)),
            ((ARRAY_U8_00, 0, 0, True, None, None), ARRAY_I8_00),
            (
                (ARRAY_F64_00, 0, 0, True, None, None),
                KeyError,
            ),  # Test on value for f64 (and f32) is not implemented
            ((ARRAY_I32_00, 0, 0, True, None, None), AssertionError),
            (
                (
                    ARRAY_F32_00,
                    0,
                    10,
                    True,
                    ARRAY_COND_U8,
                    ARRAY_F32_00_val_cond,
                ),
                ARRAY_F32_00_expected_cond_add_on_true_true,
            ),
            (
                (
                    ARRAY_F32_00,
                    0,
                    10,
                    False,
                    ARRAY_COND_U8,
                    ARRAY_F32_00_val_cond,
                ),
                ARRAY_F32_00_expected_cond_add_on_true_false,
            ),
            (
                (
                    ARRAY_F64_00,
                    0,
                    10,
                    True,
                    ARRAY_COND_U8,
                    ARRAY_F64_00_val_cond,
                ),
                ARRAY_F64_00_expected_cond_add_on_true_true,
            ),
        ],
    )
    @pytest.mark.parametrize("win", [None, WIN_00])
    def test_array_add(self, data, expected, win):
        """Test array_replace method"""
        array, val_cond, val_add, add_on_true, array_cond, array_val_cond = data
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
            array_add(
                array_copy, val_cond, val_add, add_on_true, array_cond, array_val_cond, window
            )
        except Exception as e:
            if isinstance(e, expected):
                pass
            else:
                raise
        else:
            assert np.all(array_copy == expected_array)

    @pytest.mark.parametrize(
        "in_type, out_type, expected",
        [
            ("int8", "int8", False),
            ("int8", "int16", False),
            ("int8", "int32", False),
            ("int8", "int64", False),
            ("int8", "uint8", True),
            ("int8", "uint16", True),
            ("int8", "uint32", True),
            ("int8", "uint64", True),
            ("int8", "float16", False),
            ("int8", "float32", False),
            ("int8", "float64", False),
            ("int16", "int8", True),
            ("int16", "int16", False),
            ("int16", "int32", False),
            ("int16", "int64", False),
            ("int16", "uint8", True),
            ("int16", "uint16", True),
            ("int16", "uint32", True),
            ("int16", "uint64", True),
            ("int16", "float16", False),
            ("int16", "float32", False),
            ("int16", "float64", False),
            ("int32", "int8", True),
            ("int32", "int16", True),
            ("int32", "int32", False),
            ("int32", "int64", False),
            ("int32", "uint8", True),
            ("int32", "uint16", True),
            ("int32", "uint32", True),
            ("int32", "uint64", True),
            ("int32", "float16", True),
            ("int32", "float32", False),
            ("int32", "float64", False),
            ("int64", "int8", True),
            ("int64", "int16", True),
            ("int64", "int32", True),
            ("int64", "int64", False),
            ("int64", "uint8", True),
            ("int64", "uint16", True),
            ("int64", "uint32", True),
            ("int64", "uint64", True),
            ("int64", "float16", True),
            ("int64", "float32", False),
            ("int64", "float64", False),
            ("uint8", "int8", True),
            ("uint8", "int16", False),
            ("uint8", "int32", False),
            ("uint8", "int64", False),
            ("uint8", "uint8", False),
            ("uint8", "uint16", False),
            ("uint8", "uint32", False),
            ("uint8", "uint64", False),
            ("uint8", "float16", False),
            ("uint8", "float32", False),
            ("uint8", "float64", False),
            ("uint16", "int8", True),
            ("uint16", "int16", True),
            ("uint16", "int32", False),
            ("uint16", "int64", False),
            ("uint16", "uint8", True),
            ("uint16", "uint16", False),
            ("uint16", "uint32", False),
            ("uint16", "uint64", False),
            ("uint16", "float16", True),
            ("uint16", "float32", False),
            ("uint16", "float64", False),
            ("uint32", "int8", True),
            ("uint32", "int16", True),
            ("uint32", "int32", True),
            ("uint32", "int64", False),
            ("uint32", "uint8", True),
            ("uint32", "uint16", True),
            ("uint32", "uint32", False),
            ("uint32", "uint64", False),
            ("uint32", "float16", True),
            ("uint32", "float32", False),
            ("uint32", "float64", False),
            ("uint64", "int8", True),
            ("uint64", "int16", True),
            ("uint64", "int32", True),
            ("uint64", "int64", True),
            ("uint64", "uint8", True),
            ("uint64", "uint16", True),
            ("uint64", "uint32", True),
            ("uint64", "uint64", False),
            ("uint64", "float16", True),
            ("uint64", "float32", False),
            ("uint64", "float64", False),
            ("float16", "int8", True),
            ("float16", "int16", True),
            ("float16", "int32", False),
            ("float16", "int64", False),
            ("float16", "uint8", True),
            ("float16", "uint16", True),
            ("float16", "uint32", True),
            ("float16", "uint64", True),
            ("float16", "float16", False),
            ("float16", "float32", False),
            ("float16", "float64", False),
            ("float32", "int8", True),
            ("float32", "int16", True),
            ("float32", "int32", True),
            ("float32", "int64", True),
            ("float32", "uint8", True),
            ("float32", "uint16", True),
            ("float32", "uint32", True),
            ("float32", "uint64", True),
            ("float32", "float16", True),
            ("float32", "float32", False),
            ("float32", "float64", False),
            ("float64", "int8", True),
            ("float64", "int16", True),
            ("float64", "int32", True),
            ("float64", "int64", True),
            ("float64", "uint8", True),
            ("float64", "uint16", True),
            ("float64", "uint32", True),
            ("float64", "uint64", True),
            ("float64", "float16", True),
            ("float64", "float32", True),
            ("float64", "float64", False),
        ],
    )
    def test_is_clip_required(self, in_type, out_type, expected):
        """Test is_clip_required function for all type combinations"""
        in_dtype = np.dtype(in_type)
        out_dtype = np.dtype(out_type)
        result = is_clip_required(in_dtype, out_dtype)
        assert result == expected

    @pytest.mark.parametrize(
        "in_type, out_type, expected",
        [
            ("int8", "int8", True),  # clipping not required
            ("int8", "int16", True),  # clipping not required
            ("int8", "int32", True),  # clipping not required
            ("int8", "int64", True),  # clipping not required
            ("int8", "uint8", True),
            ("int8", "uint16", True),
            ("int8", "uint32", True),
            ("int8", "uint64", True),
            ("int8", "float16", True),  # clipping not required
            ("int8", "float32", True),  # clipping not required
            ("int8", "float64", True),  # clipping not required
            ("int16", "int8", True),
            ("int16", "int16", True),
            ("int16", "int32", True),
            ("int16", "int64", True),
            ("int16", "uint8", True),
            ("int16", "uint16", True),
            ("int16", "uint32", True),
            ("int16", "uint64", True),
            ("int16", "float16", True),
            ("int16", "float32", True),
            ("int16", "float64", True),
            ("int32", "int8", True),
            ("int32", "int16", True),
            ("int32", "int32", True),
            ("int32", "int64", True),
            ("int32", "uint8", True),
            ("int32", "uint16", True),
            ("int32", "uint32", True),
            ("int32", "uint64", True),
            ("int32", "float16", True),
            ("int32", "float32", True),
            ("int32", "float64", True),
            ("int64", "int8", True),
            ("int64", "int16", True),
            ("int64", "int32", True),
            ("int64", "int64", True),
            ("int64", "uint8", True),
            ("int64", "uint16", True),
            ("int64", "uint32", True),
            ("int64", "uint64", True),
            ("int64", "float16", True),
            ("int64", "float32", True),
            ("int64", "float64", True),
            ("uint8", "int8", True),
            ("uint8", "int16", True),
            ("uint8", "int32", True),
            ("uint8", "int64", True),
            ("uint8", "uint8", True),
            ("uint8", "uint16", True),
            ("uint8", "uint32", True),
            ("uint8", "uint64", True),
            ("uint8", "float16", True),
            ("uint8", "float32", True),
            ("uint8", "float64", True),
            ("uint16", "int8", True),
            ("uint16", "int16", True),
            ("uint16", "int32", True),
            ("uint16", "int64", True),
            ("uint16", "uint8", True),
            ("uint16", "uint16", True),
            ("uint16", "uint32", True),
            ("uint16", "uint64", True),
            ("uint16", "float16", True),
            ("uint16", "float32", True),
            ("uint16", "float64", True),
            ("uint32", "int8", True),
            ("uint32", "int16", True),
            ("uint32", "int32", True),
            ("uint32", "int64", True),
            ("uint32", "uint8", True),
            ("uint32", "uint16", True),
            ("uint32", "uint32", True),
            ("uint32", "uint64", True),
            ("uint32", "float16", True),
            ("uint32", "float32", True),
            ("uint32", "float64", True),
            ("uint64", "int8", True),
            ("uint64", "int16", True),
            ("uint64", "int32", True),
            ("uint64", "int64", True),
            ("uint64", "uint8", True),
            ("uint64", "uint16", True),
            ("uint64", "uint32", True),
            ("uint64", "uint64", True),
            ("uint64", "float16", True),
            ("uint64", "float32", True),
            ("uint64", "float64", True),
            ("float16", "int8", True),
            ("float16", "int16", False),
            ("float16", "int32", True),  # clipping not required
            ("float16", "int64", True),  # clipping not required
            ("float16", "uint8", True),
            ("float16", "uint16", False),
            ("float16", "uint32", False),
            ("float16", "uint64", False),
            ("float16", "float16", True),  # clipping not required
            ("float16", "float32", True),  # clipping not required
            ("float16", "float64", True),  # clipping not required
            ("float32", "int8", True),
            ("float32", "int16", True),
            ("float32", "int32", False),
            ("float32", "int64", False),
            ("float32", "uint8", True),
            ("float32", "uint16", True),
            ("float32", "uint32", False),
            ("float32", "uint64", False),
            ("float32", "float16", True),
            ("float32", "float32", True),
            ("float32", "float64", True),
            ("float64", "int8", True),
            ("float64", "int16", True),
            ("float64", "int32", True),
            ("float64", "int64", False),
            ("float64", "uint8", True),
            ("float64", "uint16", True),
            ("float64", "uint32", True),
            ("float64", "uint64", False),
            ("float64", "float16", True),
            ("float64", "float32", True),
            ("float64", "float64", True),
        ],
    )
    def test_is_clip_to_dtype_limits_safe(self, in_type, out_type, expected):
        """Test is_clip_required function for all type combinations"""
        in_dtype = np.dtype(in_type)
        out_dtype = np.dtype(out_type)
        result = is_clip_to_dtype_limits_safe(in_dtype, out_dtype)
        assert result == expected

    @pytest.mark.parametrize(
        "in_type, out_type, clip, safe, rounding_method, expected",
        [
            # Float 64 to int conversions
            ("float64", "int32", "auto", True, "round", ARRAY_CONVERT_DATA["int32"]),
            ("float64", "uint32", "auto", True, "round", ARRAY_CONVERT_DATA["uint32"]),
            ("float64", "int16", "auto", True, "round", ARRAY_CONVERT_DATA["int16"]),
            ("float64", "uint16", "auto", True, "round", ARRAY_CONVERT_DATA["uint16"]),
            ("float64", "int8", "auto", True, "round", ARRAY_CONVERT_DATA["int8"]),
            ("float64", "uint8", "auto", True, "round", ARRAY_CONVERT_DATA["uint8"]),
            # Float 64 to int 64 conversions => should raises an exception
            ("float64", "int64", "auto", True, "round", Exception),
            ("float64", "uint64", "auto", True, "round", Exception),
            # Float 64 to other float types
            ("float64", "float64", "auto", True, "round", ARRAY_CONVERT_DATA["float64"]),
            ("float64", "float32", "auto", True, "round", ARRAY_CONVERT_DATA["float32"]),
            # Float 32 to int conversions
            ("float32", "int32", "auto", True, "round", Exception),
            ("float32", "uint32", "auto", True, "round", Exception),
            ("float32", "int16", "auto", True, "round", ARRAY_CONVERT_DATA["int16"]),
            ("float32", "uint16", "auto", True, "round", ARRAY_CONVERT_DATA["uint16"]),
            ("float32", "int8", "auto", True, "round", ARRAY_CONVERT_DATA["int8"]),
            ("float32", "uint8", "auto", True, "round", ARRAY_CONVERT_DATA["uint8"]),
            ("float32", "int64", "auto", True, "round", Exception),
            ("float32", "uint64", "auto", True, "round", Exception),
            # Float 32 to other float types
            ("float32", "float32", "auto", True, "round", ARRAY_CONVERT_DATA["float32"]),
            (
                "float32",
                "float64",
                "auto",
                True,
                "round",
                ARRAY_CONVERT_DATA["float32"].astype(np.float64),
            ),
        ],
    )
    def test_array_convert(self, in_type, out_type, clip, safe, rounding_method, expected):
        """Test array_convert function for various type conversions"""
        array_in = np.copy(ARRAY_CONVERT_DATA[in_type])
        array_out = np.empty_like(array_in, dtype=out_type)

        try:
            array_convert(array_in, array_out, clip, safe, rounding_method)
        except Exception as e:
            if isinstance(e, expected):
                pass
            else:
                raise
        else:
            assert np.allclose(array_out, expected), f"Failed for {in_type} to {out_type}"

    @pytest.mark.parametrize(
        "in_type, out_type, clip, rounding_method, expected",
        [
            (
                "float64",
                "int32",
                "auto",
                "round",
                np.array([-1, -1, 0, 0, 0, 0, 0, 1.0], dtype=np.int32),
            ),
            (
                "float64",
                "int32",
                "auto",
                "ceil",
                np.array([-1, 0, 0, 0, 0, 1, 1, 1], dtype=np.int32),
            ),
            (
                "float64",
                "int32",
                "auto",
                "floor",
                np.array([-1, -1, -1, -1, 0.0, 0, 0, 0], dtype=np.int32),
            ),
            (
                "float64",
                "int32",
                "auto",
                None,
                np.array([-1, 0, 0, 0, 0.0, 0, 0, 0], dtype=np.int32),
            ),
        ],
    )
    def test_array_convert_rounding_method(
        self, in_type, out_type, clip, rounding_method, expected
    ):
        """Test array_convert function for rounding methods"""
        array_in = np.copy(ARRAY_CONVERT_DATA_ROUNDING_METHOD[in_type])
        array_out = np.empty_like(array_in, dtype=out_type)

        try:
            array_convert(array_in, array_out, clip, True, rounding_method)
        except Exception as e:
            if isinstance(e, expected):
                pass
            else:
                raise
        else:
            assert np.allclose(array_out, expected), f"Failed for {in_type} to {out_type}"
