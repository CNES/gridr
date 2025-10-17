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
Tests for the gridr.core.interp.bspline_prefiltering module

Command to run test :
PYTHONPATH=${PWD}/python/:$PYTHONPATH pytest tests/python/gridr/core/interp/test_bspline_prefiltering.py # noqa E501
"""
import numpy as np
import pytest

from gridr.core.interp.bspline_prefiltering import (
    array_bspline_prefiltering,
    compute_2d_domain_extension,
    compute_2d_truncation_index,
    compute_bspline_total_margin,
)
from gridr.core.interp.interpolator import BSpline3Interpolator

def assert_allclose_with_details(actual, desired, rtol=1e-7, atol=1e-8, err_msg=""):
    """
    Similar to np.testing.assert_allclose but provides detailed information about
    mismatches when the test fails.

    Parameters:
    -----------
    actual : array_like
        Array obtained from computation.
    desired : array_like
        Array to compare against.
    rtol : float, optional
        Relative tolerance parameter (default is 1e-7).
    atol : float, optional
        Absolute tolerance parameter (default is 1e-8).
    err_msg : str, optional
        Error message to display if the test fails.
    """
    actual = np.asarray(actual)
    desired = np.asarray(desired)

    if actual.shape != desired.shape:
        raise AssertionError(f"Shape mismatch: actual {actual.shape} vs desired {desired.shape}")

    diff = np.abs(actual - desired)
    max_diff = np.max(diff)

    # Check if the maximum difference is within tolerance
    if max_diff <= (atol + rtol * np.max(np.abs(desired))):
        return True

    # If not, find and display the mismatches
    error_mask = diff > (atol + rtol * np.abs(desired))
    error_indices = np.argwhere(error_mask)

    error_message = err_msg + "\n"
    error_message += f"Maximum difference: {max_diff}\n"
    error_message += f"Number of elements with differences: {np.sum(error_mask)}\n"

    if len(error_indices) > 0:
        error_message += "First 10 mismatched elements and their positions:\n"
        for idx in error_indices[:10]:
            error_message += (
                f"Position {idx}: actual={actual[idx[0], idx[1]]}, "
                f"desired={desired[idx[0], idx[1]]}, "
                f"difference={diff[idx[0], idx[1]]}\n"
            )

    raise AssertionError(error_message)

class TestBSplinePrefiltering:
    """Test class"""

    @pytest.mark.parametrize(
        "order, precision, expected",
        [
            (11, 6, np.array((58, 20, 11, 7, 4), dtype=np.uintp)),
            (3, 8, np.array((17, 0, 0, 0, 0), dtype=np.uintp)),
        ],
    )
    def test_compute_2d_truncation_index(
        self,
        order,
        precision,
        expected,
    ):
        trunc_idx = compute_2d_truncation_index(order, precision)
        np.testing.assert_array_equal(trunc_idx, expected)

    @pytest.mark.parametrize(
        "order, precision, expected",
        [
            (11, 6, np.array((105, 47, 27, 16, 9, 5), dtype=np.uintp)),
            (3, 8, np.array((18, 1, 0, 0, 0, 0), dtype=np.uintp)),
        ],
    )
    def test_compute_2d_domain_extension(
        self,
        order,
        precision,
        expected,
    ):
        lext = compute_2d_domain_extension(order, precision)
        np.testing.assert_array_equal(lext, expected)

    @pytest.mark.parametrize(
        "order, precision, expected",
        [
            (11, 6, 105),
            (3, 8, 18),
        ],
    )
    def test_compute_bspline_total_margin(
        self,
        order,
        precision,
        expected,
    ):
        margin = compute_bspline_total_margin(order, precision)
        np.testing.assert_array_equal(margin, expected)

    TEST_ARRAY_BSPLINE_PREFILTERING_NO_MASK_EXPECTED = np.asarray([ 77., 76., 75., 74., 73., 72., 71., 70., 71., 72., 73., 74., 75., 76., 77., 78., 79., 78., 77., 76., 75., 74., 73., 72.,
                67., 66., 65., 64., 63., 62., 61., 60., 61., 62., 63., 64., 65., 66., 67., 68., 69., 68., 67., 66., 65., 64., 63., 62.,
                57., 56., 55., 54., 53., 52., 51., 50., 51., 52., 53., 54., 55., 56., 57., 58., 59., 58., 57., 56., 55., 54., 53., 52.,
                47., 46., 45., 44., 43., 42., 41., 40., 41., 42., 43., 44., 45., 46., 47., 48., 49., 48., 47., 46., 45., 44., 43., 42.,
                37., 36., 35., 34., 33., 32., 31., 30., 31., 32., 33., 34., 35., 36., 37., 38., 39., 38., 37., 36., 35., 34., 33., 32.,
                27., 26., 25., 24., 23., 22., 21., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 28., 27., 26., 25., 24., 23., 22.,
                3.093084, 2.926395, 2.759706, 2.593016, 2.426327, 2.259638, 0.353195, 0.304984, 0.353128, 0.375452, 0.404703, 0.432064, 0.460056, 0.487418, 0.516666, 0.539001, 0.587104, 0.539046, 3.093084, 2.926395, 2.759706, 2.593016, 2.426327, 2.259638,
                0.203903, 0.037242, -0.129419, -0.296079, -0.462740, -0.629400, -0.128372, -0.176493, -0.128379, -0.106053, -0.076809, -0.049451, -0.021465, 0.005892, 0.035137, 0.057462, 0.105579, 0.057446, 0.203903, 0.037242, -0.129419, -0.296079, -0.462740, -0.629400,
                3.091305, 2.924637, 2.757969, 2.591300, 2.424632, 2.257964, 0.352919, 0.304714, 0.352852, 0.375173, 0.404420, 0.431778, 0.459766, 0.487125, 0.516370, 0.538701, 0.586799, 0.538746, 3.091305, 2.924637, 2.757969, 2.591300, 2.424632, 2.257964,
                4.430877, 4.264211, 4.097544, 3.930878, 3.764212, 3.597546, 0.576213, 0.527970, 0.576118, 0.598436, 0.627684, 0.655041, 0.683029, 0.710388, 0.739631, 0.761965, 0.810051, 0.762038, 4.430877, 4.264211, 4.097544, 3.930878, 3.764212, 3.597546,
                6.185188, 6.018521, 5.851854, 5.685187, 5.518521, 5.351854, 0.868638, 0.820344, 0.868506, 0.890820, 0.920069, 0.947426, 0.975414, 1.002773, 1.032015, 1.054353, 1.102426, 1.054463, 6.185188, 6.018521, 5.851854, 5.685187, 5.518521, 5.351854,
                7.828372, 7.661706, 7.495039, 7.328372, 7.161706, 6.995039, 1.142539, 1.094199, 1.142373, 1.164683, 1.193933, 1.221290, 1.249278, 1.276637, 1.305879, 1.328220, 1.376280, 1.328364, 7.828372, 7.661706, 7.495039, 7.328372, 7.161706, 6.995039,
                9.501323, 9.334656, 9.167990, 9.001323, 8.834656, 8.667990, 1.421402, 1.373013, 1.421200, 1.443508, 1.472758, 1.500115, 1.528103, 1.555463, 1.584703, 1.607048, 1.655095, 1.607227, 9.501323, 9.334656, 9.167990, 9.001323, 8.834656, 8.667990,
                11.166336, 10.999669, 10.833003, 10.666336, 10.499669, 10.333003, 1.698942, 1.650506, 1.698705, 1.721009, 1.750261, 1.777618, 1.805605, 1.832965, 1.862205, 1.884553, 1.932587, 1.884767, 11.166336, 10.999669, 10.833003, 10.666336, 10.499669, 10.333003,
                12.833333, 12.666667, 12.500000, 12.333333, 12.166667, 12., 1.976812, 1.928328, 1.976541, 1.998841, 2.028094, 2.055450, 2.083438, 2.110798, 2.140037, 2.162388, 2.210410, 2.162637, 12.833333, 12.666667, 12.500000, 12.333333, 12.166667, 12.,
                14.500330, 14.333664, 14.166997, 14.000330, 13.833664, 13.666997, 2.254683, 2.206151, 2.254376, 2.276673, 2.305927, 2.333283, 2.361271, 2.388631, 2.417869, 2.440224, 2.488233, 2.440508, 14.500330, 14.333664, 14.166997, 14.000330, 13.833664, 13.666997,
                16.165345, 15.998678, 15.832012, 15.665345, 15.498678, 15.332012, 2.532223, 2.483644, 2.531882, 2.554175, 2.583429, 2.610786, 2.638773, 2.666134, 2.695371, 2.717729, 2.765725, 2.718048, 16.165345, 15.998678, 15.832012, 15.665345, 15.498678, 15.332012,
                17.838290, 17.671623, 17.504956, 17.338290, 17.171623, 17.004956, 2.811085, 2.762458, 2.810708, 2.832999, 2.862254, 2.889610, 2.917597, 2.944958, 2.974194, 2.996556, 3.044539, 2.996910, 17.838290, 17.671623, 17.504956, 17.338290, 17.171623, 17.004956,
                19.481497, 19.314830, 19.148163, 18.981497, 18.814830, 18.648163, 3.084989, 3.036315, 3.084579, 3.106866, 3.136122, 3.163477, 3.191465, 3.218826, 3.248061, 3.270426, 3.318397, 3.270815, 19.481497, 19.314830, 19.148163, 18.981497, 18.814830, 18.648163,
                21.235723, 21.069057, 20.902391, 20.735724, 20.569058, 20.402392, 3.377401, 3.328676, 3.376953, 3.399236, 3.428493, 3.455849, 3.483836, 3.511197, 3.540431, 3.562800, 3.610758, 3.563225, 21.235723, 21.069057, 20.902391, 20.735724, 20.569058, 20.402392,
                22.575611, 22.408943, 22.242275, 22.075606, 21.908938, 21.742270, 3.600744, 3.551980, 3.600268, 3.622549, 3.651807, 3.679162, 3.707150, 3.734512, 3.763746, 3.786117, 3.834065, 3.786571, 22.575611, 22.408943, 22.242275, 22.075606, 21.908938, 21.742270,
                25.461832, 25.295172, 25.128511, 24.961850, 24.795190, 24.628529, 4.081853, 4.033009, 4.081317, 4.103591, 4.132849, 4.160203, 4.188189, 4.215550, 4.244781, 4.267158, 4.315081, 4.267671, 25.461832, 25.295172, 25.128511, 24.961850, 24.795190, 24.628529,
                22.577060, 22.410370, 22.243681, 22.076992, 21.910303, 21.743613, 3.600965, 3.552195, 3.600489, 3.622773, 3.652034, 3.679393, 3.707385, 3.734750, 3.763987, 3.786362, 3.834315, 3.786815, 22.577060, 22.410370, 22.243681, 22.076992, 21.910303, 21.743613,
                127., 126., 125., 124., 123., 122., 121., 120., 121., 122., 123., 124., 125., 126., 127., 128., 129., 128., 127., 126., 125., 124., 123., 122.,
                117., 116., 115., 114., 113., 112., 111., 110., 111., 112., 113., 114., 115., 116., 117., 118., 119., 118., 117., 116., 115., 114., 113., 112.,
                107., 106., 105., 104., 103., 102., 101., 100., 101., 102., 103., 104., 105., 106., 107., 108., 109., 108., 107., 106., 105., 104., 103., 102.,
                97., 96., 95., 94., 93., 92., 91., 90., 91., 92., 93., 94., 95., 96., 97., 98., 99., 98., 97., 96., 95., 94., 93., 92.,
                87., 86., 85., 84., 83., 82., 81., 80., 81., 82., 83., 84., 85., 86., 87., 88., 89., 88., 87., 86., 85., 84., 83., 82.,
                77., 76., 75., 74., 73., 72., 71., 70., 71., 72., 73., 74., 75., 76., 77., 78., 79., 78., 77., 76., 75., 74., 73., 72.]
             ).reshape((29, 24))
    @pytest.mark.parametrize(
        "array_in, interp, n, trunc_idx, precision, expected",
        [
            (np.pad(np.arange(10*15, dtype=np.float64).reshape(15,10), 7, mode="reflect"),
             BSpline3Interpolator(1e-2, 1), None, None, None, TEST_ARRAY_BSPLINE_PREFILTERING_NO_MASK_EXPECTED,
            ),
            (np.pad(np.arange(10*15, dtype=np.float64).reshape(15,10), 7, mode="reflect"),
             BSpline3Interpolator(1e-2, 1), 3, None, None, ValueError,
            ),
            (np.pad(np.arange(10*15, dtype=np.float64).reshape(15,10), 7, mode="reflect"),
             None, None, None, None, ValueError,
            ),
            (np.pad(np.arange(10*15, dtype=np.float64).reshape(15,10), 7, mode="reflect"),
             None, 3, None, None, Exception,
            ),
            (np.pad(np.arange(10*15, dtype=np.float64).reshape(15,10), 7, mode="reflect"),
             None, 3, None, 2, TEST_ARRAY_BSPLINE_PREFILTERING_NO_MASK_EXPECTED,
            ),
            (np.pad(np.arange(10*15, dtype=np.float64).reshape(15,10), 7, mode="reflect"),
             None, 3, compute_2d_truncation_index(3, 2), 2, Exception,
            ),
            (np.pad(np.arange(10*15, dtype=np.float64).reshape(15,10), 7, mode="reflect"),
             None, 3, compute_2d_truncation_index(3, 2), None, TEST_ARRAY_BSPLINE_PREFILTERING_NO_MASK_EXPECTED,
            ),
        ],
    )
    def test_array_bspline_prefiltering_no_mask(
        self, array_in, interp, n, trunc_idx, precision, expected
    ):
        if interp is not None:
            interp.initialize()
        array_in_cp = np.copy(array_in)
        try:
            _ = array_bspline_prefiltering(
                array_in=array_in_cp,
                array_in_mask = None,
                interp = interp,
                n = n,
                trunc_idx = trunc_idx,
                precision = precision,
                mask_influence_threshold = 1,
            )
        except Exception as e:
            if expected is not None and isinstance(e, expected):
                pass
            else:
                raise
        else:
            assert_allclose_with_details(array_in_cp, expected)
    
    TEST_ARRAY_BSPLINE_PREFILTERING_WITH_MASK_EXPECTED_MASK_1 = np.ones((29,24), dtype=np.uint8)
    TEST_ARRAY_BSPLINE_PREFILTERING_WITH_MASK_EXPECTED_MASK_1[0:6,:] = 0
    TEST_ARRAY_BSPLINE_PREFILTERING_WITH_MASK_EXPECTED_MASK_1[-6:,:] = 0
    TEST_ARRAY_BSPLINE_PREFILTERING_WITH_MASK_EXPECTED_MASK_1[:,0:6] = 0
    TEST_ARRAY_BSPLINE_PREFILTERING_WITH_MASK_EXPECTED_MASK_1[:,-6:] = 0
        
    TEST_ARRAY_BSPLINE_PREFILTERING_WITH_MASK_2 = \
        np.asarray([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                   ).reshape((29,24)).astype(np.uint8)
                   
    TEST_ARRAY_BSPLINE_PREFILTERING_WITH_MASK_2_EXPECTED_OUT_0d001 = \
        np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
               ).reshape((29,24)).astype(np.uint8)
               
    @pytest.mark.parametrize(
        "array_in, array_mask_in, interp, expected, expected_mask",
        [
            (np.pad(np.arange(10*15, dtype=np.float64).reshape(15,10), 7, mode="reflect"),
             np.ones((29,24), dtype=np.uint8),
             BSpline3Interpolator(1e-2, 1),
             TEST_ARRAY_BSPLINE_PREFILTERING_NO_MASK_EXPECTED,
             TEST_ARRAY_BSPLINE_PREFILTERING_WITH_MASK_EXPECTED_MASK_1
            ),
            (np.pad(np.arange(10*15, dtype=np.float64).reshape(15,10), 7, mode="reflect"),
             np.ones((29,24), dtype=np.int8), # Not uint8 => AssertionError
             BSpline3Interpolator(1e-2, 1),
             AssertionError,
             None
            ),
            (np.pad(np.arange(10*15, dtype=np.float64).reshape(15,10), 7, mode="reflect"),
             np.ones((29,24), dtype=np.uint8),
             BSpline3Interpolator(1e-2, 0.01),
             TEST_ARRAY_BSPLINE_PREFILTERING_NO_MASK_EXPECTED,
             TEST_ARRAY_BSPLINE_PREFILTERING_WITH_MASK_EXPECTED_MASK_1
            ),
            (np.pad(np.arange(10*15, dtype=np.float64).reshape(15,10), 7, mode="reflect"),
             TEST_ARRAY_BSPLINE_PREFILTERING_WITH_MASK_2,
             BSpline3Interpolator(1e-2, 1),
             TEST_ARRAY_BSPLINE_PREFILTERING_NO_MASK_EXPECTED,
             TEST_ARRAY_BSPLINE_PREFILTERING_WITH_MASK_EXPECTED_MASK_1 \
                 * TEST_ARRAY_BSPLINE_PREFILTERING_WITH_MASK_2,
            ),
            (np.pad(np.arange(10*15, dtype=np.float64).reshape(15,10), 7, mode="reflect"),
             TEST_ARRAY_BSPLINE_PREFILTERING_WITH_MASK_2,
             BSpline3Interpolator(1e-2, 0.001),
             TEST_ARRAY_BSPLINE_PREFILTERING_NO_MASK_EXPECTED,
             TEST_ARRAY_BSPLINE_PREFILTERING_WITH_MASK_2_EXPECTED_OUT_0d001,
            ),
            (np.pad(np.arange(10*15, dtype=np.float64).reshape(15,10), 7, mode="reflect"),
             TEST_ARRAY_BSPLINE_PREFILTERING_WITH_MASK_2,
             None, # swap on internal n=3, precision=2, mask_influence=0.001
             TEST_ARRAY_BSPLINE_PREFILTERING_NO_MASK_EXPECTED,
             TEST_ARRAY_BSPLINE_PREFILTERING_WITH_MASK_2_EXPECTED_OUT_0d001,
            ),
        ],
    )
    def test_array_bspline_prefiltering_with_mask(
        self, array_in, array_mask_in, interp, expected, expected_mask
    ):
        n=None
        precision=None
        mask_influence_threshold=None
        if interp is not None:
            interp.initialize()
        else:
            n=3
            precision=2
            mask_influence_threshold=0.001
        array_in_cp = np.copy(array_in)
        array_mask_in_cp = np.copy(array_mask_in)
        try:
            _ = array_bspline_prefiltering(
                array_in=array_in_cp,
                array_in_mask = array_mask_in_cp,
                interp = interp,
                n = n,
                trunc_idx = None,
                precision = precision,
                mask_influence_threshold = mask_influence_threshold,
            )
        except Exception as e:
            if expected is not None and isinstance(e, expected):
                pass
            else:
                raise
        else:
            if isinstance(expected, Exception):
                raise Exception(f"Should have raised an Exception {expected!r}")
            assert_allclose_with_details(array_in_cp, expected)
            assert_allclose_with_details(array_mask_in_cp, expected_mask)
    