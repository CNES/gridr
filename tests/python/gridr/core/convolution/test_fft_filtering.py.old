# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of GRIDR
# (see https://gitlab.cnes.fr/gridr/gridr).
#
#
"""
Tests for the gridr.core.convolution.fft_filtering module

PYTHONPATH=$PWD/python:$PYTHONPATH pytest tests/python/gridr/core/convolution/test_fft_filtering.py
"""
import inspect
import numpy as np
import pytest

from gridr.core.convolution.fft_filtering import (
    BoundaryPad,
    ConvolutionOutputMode,
    normalize_axes,
    get_filter_margin,
    fft_array_filter,
    fft_array_filter_check_data,
    fft_array_filter_output_shape,
    fft_odd_filter,
    normalize_zoom_arg,
    zoom_is_supported,
    decimated_size,
)

IDENTITY_KERNEL = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
GAUSSIAN_BLUR_3_3 = 1.0 / 16.0 * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])

_ASSERT_EQUAL_STRICT_PARAM_SUPPORTED = "strict" in inspect.signature(np.testing.assert_equal).parameters
_ASSERT_ALLCLOSE_STRICT_PARAM_SUPPORTED = "strict" in inspect.signature(np.testing.assert_allclose).parameters

def assert_equal(actual, desired, err_msg="", verbose=True, *, strict=False):
    """Like numpy.testing.assert_equal, with a `strict` option (shape + dtype)
    that works whether or not the installed Numpy natively supports it."""
    if _ASSERT_EQUAL_STRICT_PARAM_SUPPORTED:
        np.testing.assert_equal(
            actual, desired, err_msg=err_msg, verbose=verbose, strict=strict
        )
        return
    
    if strict:
        actual_arr = np.asanyarray(actual)
        desired_arr = np.asanyarrany(desired)
        
        if actual_arr.shape != desired_arr.shape:
            msg = f"Shapes do not match: {actual_arr.shape} != {desired_arr.shape}"
            raise AssertionError(f"{msg}\n{err_msg}" if err_msg else msg)
        
        if actual_arr.dtype != desired_arr.dtype:
            msg = f"Dtypes do not match: {actual_arr.dtype} != {desired_arr.dtype}"
            raise AssertionError(f"{msg}\n{err_msg}" if err_msg else msg)
    
    np.testing.assert_equal(actual, desired, err_msg=err_msg, verbose=verbose)
        

def assert_allclose(
    actual,
    desired,
    rtol=1e-7,
    atol=0,
    equal_nan=True,
    err_msg="",
    verbose=True,
    *,
    strict=False
):
    """Like numpy.testing.assert_allclose, with a `strict` option (shape + dtype)
    that works whether or not the installed Numpy natively supports it."""
    if _ASSERT_ALLCLOSE_STRICT_PARAM_SUPPORTED:
        np.testing.assert_allclose(
            actual,
            desired,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
            err_msg=err_msg,
            verbose=verbose,
            strict=strict,
        )
        return
    
    if strict:
        actual_arr = np.asanyarray(actual)
        desired_arr = np.asanyarrany(desired)
        
        if actual_arr.shape != desired_arr.shape:
            msg = f"Shapes do not match: {actual_arr.shape} != {desired_arr.shape}"
            raise AssertionError(f"{msg}\n{err_msg}" if err_msg else msg)
        
        if actual_arr.dtype != desired_arr.dtype:
            msg = f"Dtypes do not match: {actual_arr.dtype} != {desired_arr.dtype}"
            raise AssertionError(f"{msg}\n{err_msg}" if err_msg else msg)
    
    np.testing.assert_allclose(
            actual,
            desired,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
            err_msg=err_msg,
            verbose=verbose,
        )


class TestFFTFiltering:
    """Test class"""

    def test_fft_filtering_identity_1(self):
        """Test the fft_filtering_identity method"""
        nrow, ncol = 50, 60
        input_data = np.arange(nrow * ncol, dtype=np.float32).reshape((nrow, ncol))

        # Test a simple filtering with no border mode
        out1, origin1 = fft_array_filter(
            arr=input_data,
            fil=IDENTITY_KERNEL,
            win=None,
            boundary=BoundaryPad.NONE,
            out_mode=ConvolutionOutputMode.SAME,
        )

        # check that shape is the same
        assert np.all(out1.shape == input_data.shape)
        # assert origin is at 1, 1 for the kernel
        assert np.all(origin1[:, 0] == np.array(IDENTITY_KERNEL.shape) // 2)
        # assert that valid data is close
        np.testing.assert_allclose(out1[1:-1, 1:-1], input_data[1:-1, 1:-1], rtol=1e-5, atol=0)

        # Change the out_mode
        out2, origin2 = fft_array_filter(
            arr=input_data,
            fil=IDENTITY_KERNEL,
            win=None,
            boundary=BoundaryPad.NONE,
            out_mode=ConvolutionOutputMode.FULL,
        )
        assert out2.shape[0] == input_data.shape[0] + 2 * (IDENTITY_KERNEL.shape[0] // 2)
        assert out2.shape[1] == input_data.shape[1] + 2 * (IDENTITY_KERNEL.shape[1] // 2)
        assert np.all(origin2[:, 0] == np.array(IDENTITY_KERNEL.shape) // 2)
        np.testing.assert_allclose(out2[2:-2, 2:-2], input_data[1:-1, 1:-1], rtol=1e-5, atol=0)

        # Change the padding mode
        out3, origin3 = fft_array_filter(
            arr=input_data,
            fil=IDENTITY_KERNEL,
            win=None,
            boundary=(
                (BoundaryPad.REFLECT, BoundaryPad.REFLECT),
                (BoundaryPad.REFLECT, BoundaryPad.REFLECT),
            ),
            out_mode=ConvolutionOutputMode.FULL,
        )
        assert out3.shape[0] == input_data.shape[0] + 4 * (IDENTITY_KERNEL.shape[0] // 2)
        assert out3.shape[1] == input_data.shape[1] + 4 * (IDENTITY_KERNEL.shape[1] // 2)

        # Change the padding mode
        out4, origin4 = fft_array_filter(
            arr=input_data,
            fil=IDENTITY_KERNEL,
            win=None,
            boundary=(
                (BoundaryPad.REFLECT, BoundaryPad.NONE),
                (BoundaryPad.NONE, BoundaryPad.REFLECT),
            ),
            out_mode=ConvolutionOutputMode.FULL,
        )
        assert out4.shape[0] == input_data.shape[0] + 3 * (IDENTITY_KERNEL.shape[0] // 2)
        assert out4.shape[1] == input_data.shape[1] + 3 * (IDENTITY_KERNEL.shape[1] // 2)
        assert origin4[0][0] == 2 * (IDENTITY_KERNEL.shape[0] // 2)
        assert origin4[1][0] == IDENTITY_KERNEL.shape[1] // 2

        # Change the padding mode
        out5, origin5 = fft_array_filter(
            arr=input_data,
            fil=IDENTITY_KERNEL,
            win=None,
            boundary=(
                (BoundaryPad.REFLECT, BoundaryPad.NONE),
                (BoundaryPad.NONE, BoundaryPad.REFLECT),
            ),
            out_mode=ConvolutionOutputMode.SAME,
        )
        assert np.all(
            origin5[:, 0]
            == np.array([IDENTITY_KERNEL.shape[0] // 2, 0]) + np.array(IDENTITY_KERNEL.shape) // 2
        )

    def test_fft_filtering_identity_2(self):
        nrow, ncol = 50, 60
        input_data = np.arange(nrow * ncol, dtype=np.float32).reshape((nrow, ncol))

        out1, origin1 = fft_array_filter(
            arr=input_data,
            fil=IDENTITY_KERNEL,
            win=((10, 20), (30, 40)),
            boundary=BoundaryPad.NONE,
            out_mode=ConvolutionOutputMode.SAME,
        )

        # check that shape is the same
        assert np.all(out1.shape == (11, 11))
        # assert origin is at 1, 1 for the kernel
        assert np.all(origin1[:, 0] == np.array(IDENTITY_KERNEL.shape) // 2)
        # assert that valid data is close
        np.testing.assert_allclose(out1[1:-1, 1:-1], input_data[11:20, 31:40], rtol=1e-5, atol=0)

        out2, origin2 = fft_array_filter(
            arr=input_data,
            fil=IDENTITY_KERNEL,
            win=((10, 20), (30, 40)),
            boundary=BoundaryPad.NONE,
            out_mode=ConvolutionOutputMode.FULL,
        )

        # check that shape is the same
        assert np.all(
            out2.shape
            == (11 + 2 * (IDENTITY_KERNEL.shape[0] // 2), 11 + 2 * (IDENTITY_KERNEL.shape[1] // 2))
        )
        # check that origin2 windows has the right size
        assert origin2[0, 0] == IDENTITY_KERNEL.shape[1] // 2
        assert origin2[0, 1] == 10 + origin2[0, 0]
        assert origin2[1, 0] == IDENTITY_KERNEL.shape[1] // 2
        assert origin2[1, 1] == 10 + IDENTITY_KERNEL.shape[1] // 2
        assert np.all(origin2[:, 1] - origin2[:, 0] + 1 == np.asarray([11, 11]))
        # assert origin is at 1, 1 for the kernel
        assert np.all(origin2[:, 0] == np.array(IDENTITY_KERNEL.shape) // 2)
        assert np.all(origin2[:, 1] == 10 + np.array(IDENTITY_KERNEL.shape) // 2)
        # assert that valid data is close
        np.testing.assert_allclose(
            out2[
                1 + origin2[0, 0] : 1 + origin2[0, 1] - 1, 1 + origin2[1, 0] : 1 + origin2[1, 1] - 1
            ],
            input_data[11:20, 31:40],
            rtol=1e-5,
            atol=0,
        )


    @pytest.mark.parametrize(
        "axes, ndim, expected",
        [
            (None, 3, [0, 1, 2]),
            ((1, 2), 3, (1, 2)),
            ((1, -1), 4, (1, 3)),
        ]
    )
    def test_normalize_axes(self, axes, ndim, expected):
        """
        """
        if isinstance(expected, type) and issubclass(expected, Exception):
            with pytest.raises(expected):
                normalize_axes(axes, ndim)
        else:
            ret_axes = normalize_axes(axes, ndim)
            
            err_msg = (
                f"axes={axes} ndim={ndim}"
            )
            assert_equal(
                np.asarray(ret_axes), np.asarray(expected), err_msg=err_msg, strict=True
            )

    @pytest.mark.parametrize(
        "fil, zoom, ndim, axes, expected",
        [
            (np.zeros((3, 3)), (1, 1), 2, None, (1, 1)),
            (np.zeros((3, 3)), (1, 1), 2, (0, 1), (1, 1)),
            (np.zeros((3, 3)), (1, 1), 2, (0, ), (1, 0)),
            (np.zeros((3, 3)), (1, 1), 2, (1, ), (0, 1)),
            (np.zeros((1, 3, 3)), (1, 1), 3, (1, 2), (0, 1, 1)),
            (np.zeros((1, 3, 3)), (1, 1), 3, (0, 1, 2), (0, 1, 1)),
            (np.zeros((1, 3, 3)), (1, 1), 3, None, (0, 1, 1)),
            (np.zeros((4, 7, 9)), (1, 1), 3, None, (2, 3, 4)),
            (np.zeros((4, 7, 9)), (1, 1), 3, (1, 2), (0, 3, 4)),
            (np.zeros((4, 7, 9)), (1, 1), 3, (0, 1), (2, 3, 0)),
            (np.zeros((3, 3)), (1, 1), 2, (0, 1, 2), ValueError), # Bad axes
            (np.zeros((3, 3)), (1, 0), 2, None, ValueError), # Bad Q factor
            (np.zeros((3, 3)), (2, 1), 2, None, ValueError), # Unsupported zoom factor
        ]
    )
    def test_get_filter_margin(self, fil, zoom, ndim, axes, expected):
        """Test get_filter_margin.
        """
        if isinstance(expected, type) and issubclass(expected, Exception):
            with pytest.raises(expected):
                get_filter_margin(fil, zoom, ndim, axes)
        else:
            margins = get_filter_margin(fil, zoom, ndim, axes)
            
            err_msg = (
                f"fil.shape={fil.shape} zoom={zoom} ndim={ndim} axes={axes}"
            )
            assert_equal(
                np.asarray(margins), np.asarray(expected), err_msg=err_msg, strict=True
            )


    @pytest.mark.parametrize(
        "arr, fil, win, zoom, axes, expected",
        [
            (
                np.empty((30,30)),
                np.zeros((3, 3)),
                None,
                (1, 1),
                None,
                (
                    np.zeros((3, 3)), # returned fil
                    ((0, 29), (0, 29)), # returned win
                    [0, 1], # returned axes
                    (1, 1), # returned margins
                )
            ),
            # 3D data and 2D filter - adresses with axes = [1, 2]
            (
                np.empty((1, 30,30)),
                np.zeros((3, 3)),
                None,
                (1, 1),
                [1, 2], # axes
                (
                    np.zeros((1, 3, 3)), # returned fil
                    ((None, None), (0, 29), (0, 29)), # returned win
                    [1, 2], # returned axes
                    (0, 1, 1), # returned margins
                )
            ),      
        ]
    )
    def test_fft_array_filter_check_data_bis(
        self,
        arr,
        fil,
        win,
        zoom,
        axes,
        expected,
    ):
        """
        """
        if isinstance(expected, type) and issubclass(expected, Exception):
            with pytest.raises(expected):
                fft_array_filter_check_data(arr, fil, win, zoom, axes)
        else:
            ret = fft_array_filter_check_data(arr, fil, win, zoom, axes)
            
            # error on filter
            err_msg = (
                f"fil.shape={fil.shape} zoom={zoom} axes={axes} - error on filter"
            )
            assert_equal(
                ret[0], np.asarray(expected[0]), err_msg=err_msg, strict=True,
            )
            # error on win
            err_msg = (
                f"fil.shape={fil.shape} zoom={zoom} axes={axes} - error on win"
            )
            assert_equal(
                np.asarray(ret[1]), np.asarray(expected[1]), err_msg=err_msg, strict=True,
            )
            # error on axes
            err_msg = (
                f"fil.shape={fil.shape} zoom={zoom} axes={axes} - error on axes"
            )
            assert_equal(
                np.asarray(ret[2]), np.asarray(expected[2]), err_msg=err_msg, strict=True,
            )
            # error on conv_margins
            err_msg = (
                f"fil.shape={fil.shape} zoom={zoom} axes={axes} - error on conv_margins"
            )
            assert_equal(
                np.asarray(ret[3]), np.asarray(expected[3]), err_msg=err_msg, strict=True,
            )


    @pytest.mark.parametrize(
        "boundary_condition",
        [
             BoundaryPad.NONE,
             BoundaryPad.REFLECT,    
            ((BoundaryPad.NONE, BoundaryPad.REFLECT), (BoundaryPad.REFLECT, BoundaryPad.REFLECT)),
            ((BoundaryPad.NONE, BoundaryPad.REFLECT), (BoundaryPad.REFLECT, BoundaryPad.NONE)),
            ((BoundaryPad.NONE, BoundaryPad.REFLECT), (BoundaryPad.NONE, BoundaryPad.REFLECT)),
            ((BoundaryPad.NONE, BoundaryPad.REFLECT), (BoundaryPad.NONE, BoundaryPad.NONE)),
            ((BoundaryPad.NONE, BoundaryPad.NONE), (BoundaryPad.REFLECT, BoundaryPad.REFLECT)),
            ((BoundaryPad.NONE, BoundaryPad.NONE), (BoundaryPad.REFLECT, BoundaryPad.NONE)),
            ((BoundaryPad.NONE, BoundaryPad.NONE), (BoundaryPad.NONE, BoundaryPad.REFLECT)),
            ((BoundaryPad.NONE, BoundaryPad.NONE), (BoundaryPad.NONE, BoundaryPad.NONE)),
            ((BoundaryPad.REFLECT, BoundaryPad.REFLECT), (BoundaryPad.REFLECT, BoundaryPad.REFLECT)),
            ((BoundaryPad.REFLECT, BoundaryPad.REFLECT), (BoundaryPad.REFLECT, BoundaryPad.NONE)),
            ((BoundaryPad.REFLECT, BoundaryPad.REFLECT), (BoundaryPad.NONE, BoundaryPad.REFLECT)),
            ((BoundaryPad.REFLECT, BoundaryPad.REFLECT), (BoundaryPad.NONE, BoundaryPad.NONE)),
            ((BoundaryPad.REFLECT, BoundaryPad.NONE), (BoundaryPad.REFLECT, BoundaryPad.REFLECT)),
            ((BoundaryPad.REFLECT, BoundaryPad.NONE), (BoundaryPad.REFLECT, BoundaryPad.NONE)),
            ((BoundaryPad.REFLECT, BoundaryPad.NONE), (BoundaryPad.NONE, BoundaryPad.REFLECT)),
            ((BoundaryPad.REFLECT, BoundaryPad.NONE), (BoundaryPad.NONE, BoundaryPad.NONE)),
        ]
    )
    @pytest.mark.parametrize(
        "out_mode",
        [
            ConvolutionOutputMode.SAME,
        ]
    )
    @pytest.mark.parametrize(
        "nrow, ncol, nvar, win, zoom",
        [
            (50, 60, 1, ((10, 20), (30,42)), (1, 2)),
            (50, 60, 1, ((10, 20), (30,42)), (1, 3)),
            (50, 60, 1, ((10, 20), (30,42)), (1, 5)),
            (50, 60, 1, ((10, 20), (30,42)), (1, 50)),    
            (50, 60, 2, ((0, 1), (10, 20), (30,42)), (1, 5)),
        ]
    )
    @pytest.mark.parametrize(
        "centered_decimation",
        [
            True,
            False,
        ]
    )
    def test_fft_array_filter_q_greater_than_1(
        self, boundary_condition, out_mode, nrow, ncol, nvar, win, zoom, centered_decimation
    ):
        """
        Test the fft_array_filter with zoom Q > 1 by with an a posteriori
        decimation
        """
        axes = None
        shape = (nrow, ncol)
        if nvar > 1:
            axes = (1, 2)
            shape = (nvar, nrow, ncol)
        
        input_data = np.arange(nrow * ncol * nvar, dtype=np.float32).reshape(shape)
        
        kwargs = {
            'arr': input_data,
            'fil': IDENTITY_KERNEL,
            'win': win,
            'boundary': boundary_condition,
            'out_mode': out_mode,
            'zoom': zoom,
            'centered_decimation': centered_decimation,
            'axes': axes,
        }
        
        arr_out, _ = fft_array_filter(**kwargs)
        
        # update kwargs to set zoom Q to be equal to 1.
        Q = zoom[1]
        kwargs['zoom'] = (zoom[0], 1)
        arr_val, _ = fft_array_filter(**kwargs)
        
        offset = 0
        if centered_decimation:
            if Q % 2:
                offset = Q // 2
            else:
                offset = (Q - 1) // 2
        if nvar > 1:
            arr_val = arr_val[:, offset::Q, offset::Q]
        else:
            arr_val = arr_val[offset::Q, offset::Q]
        
        err_msg = (
            f"boundary={boundary_condition}, out_mode={out_mode}, "
            f"zoom={zoom}, centered_decimation={centered_decimation} "
            f"nrow={ncol}, ncol={ncol}, nvar={nvar}",
        )
        assert_allclose(
            arr_out, arr_val, rtol=1e-6, atol=0, err_msg=err_msg, strict=True,
        )


    def test_fft_array_filter_check_data(self):
        """Test the fft_array_filter_check_data_method"""
        # test 2d data
        nrow, ncol = 50, 60
        arr = np.arange(nrow * ncol, dtype=np.float32).reshape((nrow, ncol))

        fil_odd = np.zeros((4, 4))
        fil_even, win, axes, conv_margins = fft_array_filter_check_data(
            arr, fil=fil_odd, win=None, zoom=1, axes=None
        )

        # check the filter shape is odd
        assert np.all(fil_even.shape == (5, 5))
        # check the window is bidimensionnal
        assert win.ndim == 2
        # check the window shape matches (2,2)
        assert np.all(win.shape == (2, 2))
        # check the window covers all data
        assert np.all(win == ((0, nrow - 1), (0, ncol - 1)))
        # check axes are explicit and expected length
        assert len(axes) == arr.ndim
        # check the conv margins are ok
        assert np.all(conv_margins == [fil_even.shape[0] // 2, fil_even.shape[1] // 2])

    def test_fft_odd_filter(self):
        """Test the fft odd filter method"""
        # First check an already odd filter
        odd_filter = np.ones((5, 5))
        ret_filter = fft_odd_filter(odd_filter, None)
        assert np.all(odd_filter == ret_filter)
        # same check but only on axe 1
        ret_filter = fft_odd_filter(odd_filter, axes=(0,))
        assert np.all(odd_filter == ret_filter)

        # Check a even filter
        even_filter = np.ones((4, 4))
        ret_filter = fft_odd_filter(even_filter, None)
        assert np.all(ret_filter.shape == (5, 5))
        assert np.all(even_filter == ret_filter[0:4, 0:4])
        # check last column is 0
        assert np.all(ret_filter[:, -1] == 0)
        # check last line is 0
        assert np.all(ret_filter[-1, :] == 0)

        # same check but only on axe 1
        ret_filter = fft_odd_filter(even_filter, axes=(0,))
        assert np.all(ret_filter.shape == (5, 4))
        assert np.all(even_filter == ret_filter[0:4, 0:4])
        # check last line is 0
        assert np.all(ret_filter[-1, :] == 0)

    @pytest.mark.parametrize(
        "boundary_condition",
        [
            BoundaryPad.NONE,
            BoundaryPad.REFLECT,    
            ((BoundaryPad.NONE, BoundaryPad.REFLECT), (BoundaryPad.REFLECT, BoundaryPad.REFLECT)),
            ((BoundaryPad.NONE, BoundaryPad.REFLECT), (BoundaryPad.REFLECT, BoundaryPad.NONE)),
            ((BoundaryPad.NONE, BoundaryPad.REFLECT), (BoundaryPad.NONE, BoundaryPad.REFLECT)),
            ((BoundaryPad.NONE, BoundaryPad.REFLECT), (BoundaryPad.NONE, BoundaryPad.NONE)),
            ((BoundaryPad.NONE, BoundaryPad.NONE), (BoundaryPad.REFLECT, BoundaryPad.REFLECT)),
            ((BoundaryPad.NONE, BoundaryPad.NONE), (BoundaryPad.REFLECT, BoundaryPad.NONE)),
            ((BoundaryPad.NONE, BoundaryPad.NONE), (BoundaryPad.NONE, BoundaryPad.REFLECT)),
            ((BoundaryPad.NONE, BoundaryPad.NONE), (BoundaryPad.NONE, BoundaryPad.NONE)),
            ((BoundaryPad.REFLECT, BoundaryPad.REFLECT), (BoundaryPad.REFLECT, BoundaryPad.REFLECT)),
            ((BoundaryPad.REFLECT, BoundaryPad.REFLECT), (BoundaryPad.REFLECT, BoundaryPad.NONE)),
            ((BoundaryPad.REFLECT, BoundaryPad.REFLECT), (BoundaryPad.NONE, BoundaryPad.REFLECT)),
            ((BoundaryPad.REFLECT, BoundaryPad.REFLECT), (BoundaryPad.NONE, BoundaryPad.NONE)),
            ((BoundaryPad.REFLECT, BoundaryPad.NONE), (BoundaryPad.REFLECT, BoundaryPad.REFLECT)),
            ((BoundaryPad.REFLECT, BoundaryPad.NONE), (BoundaryPad.REFLECT, BoundaryPad.NONE)),
            ((BoundaryPad.REFLECT, BoundaryPad.NONE), (BoundaryPad.NONE, BoundaryPad.REFLECT)),
            ((BoundaryPad.REFLECT, BoundaryPad.NONE), (BoundaryPad.NONE, BoundaryPad.NONE)),
        ]
    )
    @pytest.mark.parametrize(
        "out_mode",
        [
            ConvolutionOutputMode.SAME,
            ConvolutionOutputMode.FULL,
        ]
    )
    @pytest.mark.parametrize(
        "zoom",
        [
            (1, 1),
        ]
    )
    @pytest.mark.parametrize(
        "centered_decimation",
        [
            True,
            False,
        ]
    )
    def test_fft_filtering_output_shape_consistency(
        self, boundary_condition, out_mode, zoom, centered_decimation
    ):
        """
        Test the fft_filtering_output_shape method is consistent with the 
        output of fft_array_filter.
        """
        nrow, ncol = 50, 60
        input_data = np.arange(nrow * ncol, dtype=np.float32).reshape((nrow, ncol))
        
        kwargs = {
            'arr': input_data,
            'fil': IDENTITY_KERNEL,
            'win': ((10, 20), (30, 42)),
            'boundary': boundary_condition,
            'out_mode': out_mode,
            'zoom': zoom,
            'centered_decimation': centered_decimation,
        }
        
        shape_out = fft_array_filter_output_shape(**kwargs)
        arr_out, _ = fft_array_filter(**kwargs)
        
        assert np.all(shape_out == arr_out.shape), (
                f"boundary={boundary_condition}, out_mode={out_mode}, "
                f"zoom={zoom}, centered_decimation={centered_decimation} "
                f"-> expected={arr_out.shape}, got={shape_out}"
            )


    @pytest.mark.parametrize(
        "boundary_condition",
        [
            BoundaryPad.NONE,
            BoundaryPad.REFLECT,    
            ((BoundaryPad.NONE, BoundaryPad.REFLECT), (BoundaryPad.REFLECT, BoundaryPad.REFLECT)),
            ((BoundaryPad.NONE, BoundaryPad.REFLECT), (BoundaryPad.REFLECT, BoundaryPad.NONE)),
            ((BoundaryPad.NONE, BoundaryPad.REFLECT), (BoundaryPad.NONE, BoundaryPad.REFLECT)),
            ((BoundaryPad.NONE, BoundaryPad.REFLECT), (BoundaryPad.NONE, BoundaryPad.NONE)),
            ((BoundaryPad.NONE, BoundaryPad.NONE), (BoundaryPad.REFLECT, BoundaryPad.REFLECT)),
            ((BoundaryPad.NONE, BoundaryPad.NONE), (BoundaryPad.REFLECT, BoundaryPad.NONE)),
            ((BoundaryPad.NONE, BoundaryPad.NONE), (BoundaryPad.NONE, BoundaryPad.REFLECT)),
            ((BoundaryPad.NONE, BoundaryPad.NONE), (BoundaryPad.NONE, BoundaryPad.NONE)),
            ((BoundaryPad.REFLECT, BoundaryPad.REFLECT), (BoundaryPad.REFLECT, BoundaryPad.REFLECT)),
            ((BoundaryPad.REFLECT, BoundaryPad.REFLECT), (BoundaryPad.REFLECT, BoundaryPad.NONE)),
            ((BoundaryPad.REFLECT, BoundaryPad.REFLECT), (BoundaryPad.NONE, BoundaryPad.REFLECT)),
            ((BoundaryPad.REFLECT, BoundaryPad.REFLECT), (BoundaryPad.NONE, BoundaryPad.NONE)),
            ((BoundaryPad.REFLECT, BoundaryPad.NONE), (BoundaryPad.REFLECT, BoundaryPad.REFLECT)),
            ((BoundaryPad.REFLECT, BoundaryPad.NONE), (BoundaryPad.REFLECT, BoundaryPad.NONE)),
            ((BoundaryPad.REFLECT, BoundaryPad.NONE), (BoundaryPad.NONE, BoundaryPad.REFLECT)),
            ((BoundaryPad.REFLECT, BoundaryPad.NONE), (BoundaryPad.NONE, BoundaryPad.NONE)),
        ]
    )
    @pytest.mark.parametrize(
        "out_mode",
        [
            ConvolutionOutputMode.SAME,
        ]
    )
    @pytest.mark.parametrize(
        "zoom",
        [
            (1, 2),
            (1, 3),
            (1, 5),
            (1, 50),
        ]
    )
    @pytest.mark.parametrize(
        "centered_decimation",
        [
            True,
            False,
        ]
    )
    def test_fft_filtering_output_shape_consistency_q_greater_than_1(
        self, boundary_condition, out_mode, zoom, centered_decimation
    ):
        """
        Test the fft_filtering_output_shape method is consistent with the 
        output of fft_array_filter.
        """
        nrow, ncol = 50, 60
        input_data = np.arange(nrow * ncol, dtype=np.float32).reshape((nrow, ncol))
        
        kwargs = {
            'arr': input_data,
            'fil': IDENTITY_KERNEL,
            'win': ((10, 20), (30, 42)),
            'boundary': boundary_condition,
            'out_mode': out_mode,
            'zoom': zoom,
            'centered_decimation': centered_decimation,
        }
        
        shape_out = fft_array_filter_output_shape(**kwargs)
        arr_out, _ = fft_array_filter(**kwargs)
        
        assert np.all(shape_out == arr_out.shape), (
                f"boundary={boundary_condition}, out_mode={out_mode}, "
                f"zoom={zoom}, centered_decimation={centered_decimation} "
                f"-> expected={arr_out.shape}, got={shape_out}"
            )

    
    # ---------------------------------------------------------------------------
    # Test normalize_zoom_arg
    # ---------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "zoom, expected",
        [
            (1, (1, 1)),
            (0, ValueError),
            (2, (2, 1)),
            (-1, ValueError),
            (1., TypeError),
            ((1, 1), (1, 1)),
            ((1, 2), (1, 2)),
            ((3, 5), (3,5)),
            ((2, 6), (1, 3)),
            ((1, -1), ValueError),
            ((1, 0), ValueError),
        ]
    )
    def test_normalize_zoom_arg(self, zoom, expected):
        if isinstance(expected, type) and issubclass(expected, Exception):
            with pytest.raises(expected):
                normalize_zoom_arg(zoom)
        else:
            zoom_pq = normalize_zoom_arg(zoom)
            assert zoom_pq[0] == expected[0]
            assert zoom_pq[1] == expected[1]
    
    
    # ---------------------------------------------------------------------------
    # Test zoom_is_supported
    # ---------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "zoom, expected",
        [
            ((1, 1), True),
            ((1, 2), True),
            ((1, -1), False),
            ((1, 0), False),
            ((2, 1), False),
            ((0, 1), False),
        ]
    )
    def test_zoom_is_supported(self, zoom, expected):
        assert zoom_is_supported(zoom) == expected
        
    
    # ---------------------------------------------------------------------------
    # Test decimated_size
    # ---------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "N, Q, offset",
        [
            (0, 1, 0), # empty list
            (1, 1, 0), # single element, kept
            (1, 1, 1), # offset == N
            (1, 5, 0), # Q > N
            (5, 1, 0), # Q == 1 (no decimation)
            (5, 5, 0), # Q == N
            (5, 6, 0), # Q > N
            (5, 1, 4), # offset == last valid index
            (5, 1, 5), # offset == N (exact boundary)
            (5, 2, 100), # offset far beyond N
            (5, 3, 4), # offset = last index, Q > 1
        ],
    )
    def test_decimated_size_matches_slicing(self, N, Q, offset):
        arr = list(range(N))
        assert decimated_size(N, Q, offset) == len(arr[offset::Q])

    def test_decimated_size_exhaustive(self):
        for N in range(0, 30):
            arr = list(range(N))
            for Q in range(1, 8):
                for offset in range(0, N + 3):
                    assert decimated_size(N, Q, offset) == len(arr[offset::Q])

    @pytest.mark.parametrize(
        "N, Q, offset",
        [
            (-1, 1, 0), # negative N
            (5, 0, 0), # Q == 0
            (5, -1, 0), # negative Q
            (5, 1, -1), # negative offset
        ],
    )
    def test_decimated_size_invalid_domain_raises(self, N, Q, offset):
        with pytest.raises(ValueError):
            decimated_size(N, Q, offset)
