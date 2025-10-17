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
Tests for the gridr.core.interp.interpolator module

Command to run test :
PYTHONPATH=${PWD}/python/:$PYTHONPATH pytest tests/python/gridr/core/interp/test_interpolator.py
"""
import numpy as np
import pytest

from gridr.cdylib import (
    BSpline3Interpolator,
    BSpline5Interpolator,
    BSpline7Interpolator,
    BSpline9Interpolator,
    BSpline11Interpolator,
    LinearInterpolator,
    NearestInterpolator,
    OptimizedBicubicInterpolator,
    PyInterpolatorType,
)
from gridr.core.interp.interpolator import get_interpolator, is_bspline


class TestInterpolator:
    """Test class"""

    @pytest.mark.parametrize(
        "interp, interp_args, expected",
        [
            # Set of string parameters
            ("nearest", {}, "nearest"),
            ("linear", {}, "linear"),
            ("cubic", {}, "optimized_bicubic"),
            ("bspline3", {"epsilon": 1e-6, "mask_influence_threshold": 0.01}, "bspline3"),
            ("bspline3", {"epsilon": None, "mask_influence_threshold": 0.01}, TypeError),
            ("bspline3", {"epsilon": 1e-6, "mask_influence_threshold": None}, TypeError),
            ("bspline5", {"epsilon": 1e-6, "mask_influence_threshold": 0.01}, "bspline5"),
            ("bspline7", {"epsilon": 1e-6, "mask_influence_threshold": 0.01}, "bspline7"),
            ("bspline9", {"epsilon": 1e-6, "mask_influence_threshold": 0.01}, "bspline9"),
            ("bspline11", {"epsilon": 1e-6, "mask_influence_threshold": 0.01}, "bspline11"),
            ("nearest", {"param": 0}, TypeError),
            ("bspline3", {}, TypeError),
            ("unknown", {}, Exception),
            # Set of enum parameters
            (PyInterpolatorType.Nearest, {}, "nearest"),
            (PyInterpolatorType.Linear, {}, "linear"),
            (PyInterpolatorType.OptimizedBicubic, {}, "optimized_bicubic"),
            (
                PyInterpolatorType.BSpline3,
                {"epsilon": 1e-6, "mask_influence_threshold": 0.01},
                "bspline3",
            ),
            (
                PyInterpolatorType.BSpline5,
                {"epsilon": 1e-6, "mask_influence_threshold": 0.01},
                "bspline5",
            ),
            (
                PyInterpolatorType.BSpline7,
                {"epsilon": 1e-6, "mask_influence_threshold": 0.01},
                "bspline7",
            ),
            (
                PyInterpolatorType.BSpline9,
                {"epsilon": 1e-6, "mask_influence_threshold": 0.01},
                "bspline9",
            ),
            (
                PyInterpolatorType.BSpline11,
                {"epsilon": 1e-6, "mask_influence_threshold": 0.01},
                "bspline11",
            ),
            (PyInterpolatorType.Nearest, {"param": 0}, TypeError),
            (PyInterpolatorType.BSpline3, {}, TypeError),
            # Set of object parameters
            (NearestInterpolator(), {}, "nearest"),
            (LinearInterpolator(), {}, "linear"),
            (OptimizedBicubicInterpolator(), {}, "optimized_bicubic"),
            (BSpline3Interpolator(epsilon=1e-7, mask_influence_threshold=0.01), {}, "bspline3"),
            (BSpline5Interpolator(epsilon=1e-7, mask_influence_threshold=0.01), {}, "bspline5"),
            (BSpline7Interpolator(epsilon=1e-7, mask_influence_threshold=0.01), {}, "bspline7"),
            (BSpline9Interpolator(epsilon=1e-7, mask_influence_threshold=0.01), {}, "bspline9"),
            (BSpline11Interpolator(epsilon=1e-7, mask_influence_threshold=0.01), {}, "bspline11"),
            (object(), {}, Exception),
        ],
    )
    def test_get_interpolator(
        self,
        interp,
        interp_args,
        expected,
    ):
        """
        Test the get_interpolator function with various input types.

        This test verifies that:
        1. The function returns the correct interpolator type for valid inputs
        2. The function raises the expected exceptions for invalid inputs
        3. The shortname of the returned interpolator matches the expected value

        Args:
            interp: The input to the get_interpolator function (string, enum, or object)
            interp_args: Dictionary of arguments to pass to the interpolator constructor
            expected: Either the expected shortname string or the expected exception type

        Raises:
            AssertionError: If the test fails to verify the expected behavior
        """
        if isinstance(expected, type) and issubclass(expected, BaseException):
            # Test that the expected exception is raised
            with pytest.raises(expected):
                get_interpolator(interp, **interp_args)

        else:
            # Test that no exception is raised and the shortname matches
            interp_out = get_interpolator(interp, **interp_args)
            assert interp_out.shortname() == expected, (
                f"For input {interp} with args {interp_args}, "
                f"expected shortname {expected!r}, got {interp_out.shortname()!r}"
            )

    @pytest.mark.parametrize(
        "interp, expected",
        [
            # Set of object parameters
            (NearestInterpolator(), [1] * 4),
            (LinearInterpolator(), [1] * 4),
            (OptimizedBicubicInterpolator(), [2] * 4),
            (BSpline3Interpolator(epsilon=1e-7, mask_influence_threshold=0.01), [16] * 4),
            (BSpline5Interpolator(epsilon=1e-5, mask_influence_threshold=0.01), [28] * 4),
            (BSpline7Interpolator(epsilon=1e-3, mask_influence_threshold=0.01), [37] * 4),
            (BSpline9Interpolator(epsilon=1e-12, mask_influence_threshold=0.01), [121] * 4),
            (BSpline11Interpolator(epsilon=1e-6, mask_influence_threshold=0.01), [105] * 4),
        ],
    )
    def test_interpolator_total_margins(
        self,
        interp,
        expected,
    ):
        """
        Test the total_margins() method for various interpolator types.

        This test verifies that:
        1. The initialize() method executes without errors
        2. The total_margins() method returns the expected margin values
        3. The returned margins match the expected values for each interpolator type

        Args:
            interp: An instance of an interpolator class to be tested.
                Possible values include:
                - NearestInterpolator
                - LinearInterpolator
                - OptimizedBicubicInterpolator
                - BSpline3Interpolator
                - BSpline5Interpolator
                - BSpline7Interpolator
                - BSpline9Interpolator
                - BSpline11Interpolator
            expected: The expected margin values as a list of 4 integers.
                Represents [top, bottom, left, right] margins for the interpolator.

        Raises:
            AssertionError: If the actual margins don't match the expected values.
            Exception: If the initialize() or total_margins() methods raise an unexpected exception.
        """
        interp.initialize()
        total_margins = np.asarray(interp.total_margins())
        np.testing.assert_array_equal(total_margins, expected)

    @pytest.mark.parametrize(
        "interp, expected",
        [
            # Set of object parameters
            (NearestInterpolator(), None),
            (LinearInterpolator(), None),
            (OptimizedBicubicInterpolator(), None),
            (BSpline3Interpolator(epsilon=1e-7, mask_influence_threshold=0.01), ValueError),
            (BSpline5Interpolator(epsilon=1e-5, mask_influence_threshold=0.01), ValueError),
            (BSpline7Interpolator(epsilon=1e-3, mask_influence_threshold=0.01), ValueError),
            (BSpline9Interpolator(epsilon=1e-12, mask_influence_threshold=0.01), ValueError),
            (BSpline11Interpolator(epsilon=1e-6, mask_influence_threshold=0.01), ValueError),
        ],
    )
    def test_interpolator_total_margins_no_initialize(
        self,
        interp,
        expected,
    ):
        """
        Test the total_margins() method without calling initialize() first.

        This test verifies that:
        1. For some interpolators, total_margins() can be called without initialize()
        2. For other interpolators, total_margins() raises ValueError when called without
           initialize()

        Args:
            interp: An instance of an interpolator class to be tested.
            expected: The expected outcome of the test.
                - None: No exception expected
                - ValueError: ValueError exception expected

        Raises:
            AssertionError: If the actual outcome doesn't match the expected one
            ValueError: If the test fails to verify the expected exception

        Notes:
            - For interpolators where None is expected, the test verifies that total_margins()
              can be called without initialize() and does not raise any Exception.
            - For interpolators where ValueError is expected, the test verifies that
              calling total_margins() without initialize() raises the expected exception
        """
        if expected is None:
            # Test that no exception is raised
            _ = np.asarray(interp.total_margins())
        else:
            # Test that the expected exception is raised
            with pytest.raises(expected) as exc_info:  # noqa: B908
                _ = np.asarray(interp.total_margins())

                if not isinstance(exc_info.value, expected):
                    pytest.fail(
                        f"Expected exception of type {expected.__name__}, "
                        f"but got {type(exc_info.value).__name__} instead. "
                        f"Exception message: {str(exc_info.value)}"
                    )

    @pytest.mark.parametrize(
        "interp, expected",
        [
            # Set of string parameters
            ("nearest", False),
            ("linear", False),
            ("cubic", False),
            ("bspline3", True),
            ("bspline5", True),
            ("bspline7", True),
            ("bspline9", True),
            ("bspline11", True),
            ("unknown", False),
            # Set of enum parameters
            (PyInterpolatorType.Nearest, False),
            (PyInterpolatorType.Linear, False),
            (PyInterpolatorType.OptimizedBicubic, False),
            (PyInterpolatorType.BSpline3, True),
            (PyInterpolatorType.BSpline5, True),
            (PyInterpolatorType.BSpline7, True),
            (PyInterpolatorType.BSpline9, True),
            (PyInterpolatorType.BSpline11, True),
            # Set of class parameters
            (NearestInterpolator, False),
            (LinearInterpolator, False),
            (OptimizedBicubicInterpolator, False),
            (BSpline3Interpolator, True),
            (BSpline5Interpolator, True),
            (BSpline7Interpolator, True),
            (BSpline9Interpolator, True),
            (BSpline11Interpolator, True),
            (object, False),
            # Set of class parameters
            (NearestInterpolator(), False),
            (LinearInterpolator(), False),
            (OptimizedBicubicInterpolator(), False),
            (BSpline3Interpolator(epsilon=1e-6, mask_influence_threshold=0.01), True),
            (BSpline5Interpolator(epsilon=1e-6, mask_influence_threshold=0.01), True),
            (BSpline7Interpolator(epsilon=1e-6, mask_influence_threshold=0.01), True),
            (BSpline9Interpolator(epsilon=1e-6, mask_influence_threshold=0.01), True),
            (BSpline11Interpolator(epsilon=1e-6, mask_influence_threshold=0.01), True),
        ],
    )
    def test_is_bspline(
        self,
        interp,
        expected,
    ):
        """
        Test the is_bspline function with various input types.

        This test verifies that:
        1. The function returns True for all B-Spline interpolators
        2. The function returns False for all input that are no B-Spline

        Args:
            interp: The input to check if it is a B-Spline interpolator
            expected: The expected bool result
        """
        assert is_bspline(interp) is expected
