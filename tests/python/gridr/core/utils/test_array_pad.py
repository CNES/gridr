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
#
# Portions of this code are derived from NumPy's numpy.pad implementation
# Copyright (c) 2005-2025, NumPy Developers
# All rights reserved.
"""
Tests for the gridr.core.utils.array_pad module

Command to run test :
PYTHONPATH=${PWD}/python/:$PYTHONPATH pytest tests/python/gridr/core/utils/test_array_pad.py
"""
import warnings

import numpy as np
import pytest

from gridr.core.utils.array_pad import pad_inplace, pad_inplace_fallback


class TestArrayPad:

    @pytest.mark.parametrize(
        "array, src_win, pad_width",
        [
            (np.arange(10), (slice(None, None),), (0, 0)),
            (np.arange(10), (slice(None, None),), 0),
            (np.arange(10), (slice(2, 7),), (2, 3)),
            (np.arange(10), (slice(2, 8),), 2),
            (
                np.arange(20).reshape((4, 5)),
                (slice(None, None), slice(None, None)),
                ((0, 0), (0, 0)),
            ),
            (np.arange(20).reshape((4, 5)), (slice(1, -2), slice(3, 4)), ((1, 2), (3, 1))),
            (np.arange(20).reshape((4, 5)), (slice(1, -2), slice(1, -2)), (1, 2)),
            (
                np.arange(40).reshape((2, 4, 5)),
                (slice(None, None), slice(None, None), slice(None, None)),
                0,
            ),
        ],
    )
    @pytest.mark.parametrize("mode", ["constant", "edge", "reflect", "symmetric", "wrap"])
    @pytest.mark.parametrize("strict_size", [True, False])
    def test_pad_inplace_nominal(self, array, src_win, pad_width, mode, strict_size):
        """Test pad_inplace with various modes, dimensions and strict_size values."""
        # Create copy for reference
        original_array = array.copy()
        original_array_fallback = array.copy()
        reference_array = array[src_win].copy()

        # Test with numpy.pad for reference
        try:
            if mode == "constant":
                expected = np.pad(reference_array, pad_width, mode=mode, constant_values=0)
            else:
                expected = np.pad(reference_array, pad_width, mode=mode)
        except Exception:
            # Skip if numpy doesn't support this combination
            pytest.skip(f"NumPy doesn't support {mode} padding for this configuration")

        # Apply our inplace padding
        if mode == "constant":
            pad_inplace(
                original_array,
                src_win,
                pad_width,
                mode=mode,
                strict_size=strict_size,
                constant_values=0,
            )
            pad_inplace_fallback(
                original_array_fallback,
                src_win,
                pad_width,
                mode=mode,
                strict_size=strict_size,
                constant_values=0,
            )
        else:
            pad_inplace(original_array, src_win, pad_width, mode=mode, strict_size=strict_size)
            pad_inplace_fallback(
                original_array_fallback, src_win, pad_width, mode=mode, strict_size=strict_size
            )

        # Check shapes match
        assert original_array.shape == expected.shape
        # Check values match
        np.testing.assert_allclose(
            original_array, expected, rtol=1e-5, err_msg=f"Failed for mode={mode}"
        )
        # Check fallback
        np.testing.assert_allclose(
            original_array_fallback, expected, rtol=1e-5, err_msg=f"Fallback failed for mode={mode}"
        )

    @pytest.mark.parametrize("strict_size", [False, True])
    def test_pad_inplace_strict_size_exceptions(self, strict_size):
        """Test that appropriate exceptions are raised for mismatched sizes."""
        # Create array that's too small for padding
        original_array = np.arange(12)  # Length 10
        original_array_fallback = original_array.copy()
        src_win = (slice(2, 5),)
        pad_width = [(2, 4)]  # Need 3 + 2 + 4 = 9 elements, but array has 10
        mode = "constant"
        # Explicitely define the expected. Array must remain unchanged in no
        # padded regions
        expected = original_array.copy()
        expected[0:2] = 0
        expected[5 : 5 + 4] = 0

        if strict_size:
            # Should raise ValueError
            with pytest.raises(ValueError):
                pad_inplace(original_array, src_win, pad_width, mode=mode, strict_size=strict_size)

            # Should raise ValueError
            with pytest.raises(ValueError):
                pad_inplace_fallback(
                    original_array_fallback,
                    src_win,
                    pad_width,
                    mode="constant",
                    strict_size=strict_size,
                )
        else:
            # Should issue warning but not raise
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                pad_inplace(original_array, src_win, pad_width, mode=mode, strict_size=strict_size)
                pad_inplace_fallback(
                    original_array_fallback,
                    src_win,
                    pad_width,
                    mode="constant",
                    strict_size=strict_size,
                )
                assert len(w) >= 1
                assert issubclass(w[0].category, UserWarning)
                np.testing.assert_allclose(
                    original_array, expected, rtol=1e-5, err_msg=f"Fallback failed for mode={mode}"
                )
                # Check fallback
                np.testing.assert_allclose(
                    original_array_fallback,
                    expected,
                    rtol=1e-5,
                    err_msg=f"Fallback failed for mode={mode}",
                )

    @pytest.mark.parametrize("strict_size", [False, True])
    def test_pad_inplace_with_padding_larger_than_data(self, strict_size):
        """Test that appropriate exceptions are raised for mismatched sizes."""
        # Create array that's too small for padding
        original_array = np.arange(8)  # Length 8
        original_array_fallback = original_array.copy()
        src_win = (slice(2, 5),)
        pad_width = [(2, 4)]  # Need 3 + 2 + 4 = 9 elements, but array has 8

        # Should raise ValueError
        with pytest.raises(ValueError):
            pad_inplace(
                original_array, src_win, pad_width, mode="constant", strict_size=strict_size
            )
        # Should raise ValueError
        with pytest.raises(ValueError):
            pad_inplace_fallback(
                original_array_fallback,
                src_win,
                pad_width,
                mode="constant",
                strict_size=strict_size,
            )
