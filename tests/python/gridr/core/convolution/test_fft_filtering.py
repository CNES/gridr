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
"""
import os
import numpy as np
import pytest

from gridr.core.convolution.fft_filtering import (
        fft_array_filter,
        fft_odd_filter,
        fft_array_filter_check_data,
        fft_array_filter_output_shape,
        BoundaryPad,
        ConvolutionOutputMode)

IDENTITY_KERNEL = np.array([[0,0,0], [0, 1, 0], [0, 0, 0]])
GAUSSIAN_BLUR_3_3 = 1./16. * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])

class TestFFTFiltering:
    """Test class"""
    
    def test_fft_filtering_identity_1(self):
        """Test the fft_filtering_identity method
        """
        nrow, ncol = 50, 60
        input_data = np.arange(nrow*ncol, dtype=np.float32).reshape((nrow, ncol))
        
        # Test a simple filtering with no border mode
        out1, origin1 = fft_array_filter(
                arr=input_data,
                fil=IDENTITY_KERNEL,
                win=None,
                boundary=BoundaryPad.NONE,
                out_mode=ConvolutionOutputMode.SAME,)
        
        # check that shape is the same
        assert(np.all(out1.shape == input_data.shape))
        # assert origin is at 1, 1 for the kernel
        assert(np.all(origin1[:,0] == np.array(IDENTITY_KERNEL.shape) // 2))
        # assert that valid data is close
        np.testing.assert_allclose(out1[1:-1, 1:-1], input_data[1:-1,1:-1], rtol=1e-5, atol=0)
        
        # Change the out_mode
        out2, origin2 = fft_array_filter(
                arr=input_data,
                fil=IDENTITY_KERNEL,
                win=None,
                boundary=BoundaryPad.NONE,
                out_mode=ConvolutionOutputMode.FULL,)
        assert(out2.shape[0] == input_data.shape[0] + 2 * (IDENTITY_KERNEL.shape[0] // 2))
        assert(out2.shape[1] == input_data.shape[1] + 2 * (IDENTITY_KERNEL.shape[1] // 2))
        assert(np.all(origin2[:,0] == np.array(IDENTITY_KERNEL.shape) // 2))
        np.testing.assert_allclose(out2[2:-2, 2:-2], input_data[1:-1,1:-1], rtol=1e-5, atol=0)
        
        # Change the padding mode
        out3, origin3 = fft_array_filter(
                arr=input_data,
                fil=IDENTITY_KERNEL,
                win=None,
                boundary=((BoundaryPad.REFLECT, BoundaryPad.REFLECT),
                        (BoundaryPad.REFLECT, BoundaryPad.REFLECT)),
                out_mode=ConvolutionOutputMode.FULL,)
        assert(out3.shape[0] == input_data.shape[0] + 4 * (IDENTITY_KERNEL.shape[0] // 2))
        assert(out3.shape[1] == input_data.shape[1] + 4 * (IDENTITY_KERNEL.shape[1] // 2))
        
        # Change the padding mode
        out4, origin4 = fft_array_filter(
                arr=input_data,
                fil=IDENTITY_KERNEL,
                win=None,
                boundary=((BoundaryPad.REFLECT, BoundaryPad.NONE),
                        (BoundaryPad.NONE, BoundaryPad.REFLECT)),
                out_mode=ConvolutionOutputMode.FULL,)
        assert(out4.shape[0] == input_data.shape[0] + 3 * (IDENTITY_KERNEL.shape[0] // 2))
        assert(out4.shape[1] == input_data.shape[1] + 3 * (IDENTITY_KERNEL.shape[1] // 2))
        assert(origin4[0][0] == 2 * (IDENTITY_KERNEL.shape[0] // 2))
        assert(origin4[1][0] == IDENTITY_KERNEL.shape[1] // 2)
        
        # Change the padding mode
        out5, origin5 = fft_array_filter(
                arr=input_data,
                fil=IDENTITY_KERNEL,
                win=None,
                boundary=((BoundaryPad.REFLECT, BoundaryPad.NONE),
                        (BoundaryPad.NONE, BoundaryPad.REFLECT)),
                out_mode=ConvolutionOutputMode.SAME,)
        assert(np.all(origin5[:,0] == np.array([IDENTITY_KERNEL.shape[0] // 2, 0]) + np.array(IDENTITY_KERNEL.shape) // 2))
        
        
    def test_fft_filtering_identity_2(self):
        nrow, ncol = 50, 60
        input_data = np.arange(nrow*ncol, dtype=np.float32).reshape((nrow, ncol))

        out1, origin1 = fft_array_filter(
                arr=input_data,
                fil=IDENTITY_KERNEL,
                win=((10,20), (30,40)),
                boundary=BoundaryPad.NONE,
                out_mode=ConvolutionOutputMode.SAME,)
        
        # check that shape is the same
        assert(np.all(out1.shape == (11, 11)))
        # assert origin is at 1, 1 for the kernel
        assert(np.all(origin1[:,0] == np.array(IDENTITY_KERNEL.shape) // 2))
        # assert that valid data is close
        np.testing.assert_allclose(out1[1:-1, 1:-1], input_data[11:20, 31:40], rtol=1e-5, atol=0)

        out2, origin2 = fft_array_filter(
                arr=input_data,
                fil=IDENTITY_KERNEL,
                win=((10,20), (30,40)),
                boundary=BoundaryPad.NONE,
                out_mode=ConvolutionOutputMode.FULL,)
        
        # check that shape is the same
        assert(np.all(out2.shape == (11+2 * (IDENTITY_KERNEL.shape[0] // 2), 11+ 2 * (IDENTITY_KERNEL.shape[1] // 2))))
        # check that origin2 windows has the right size
        assert(origin2[0,0] == IDENTITY_KERNEL.shape[1] // 2)
        assert(origin2[0,1] == 10 + origin2[0,0])
        assert(origin2[1,0] == IDENTITY_KERNEL.shape[1] // 2)
        assert(origin2[1,1] == 10 + IDENTITY_KERNEL.shape[1] // 2)
        assert(np.all(origin2[:,1]-origin2[:,0] + 1 == np.asarray([11, 11])))
        # assert origin is at 1, 1 for the kernel
        assert(np.all(origin2[:,0] == np.array(IDENTITY_KERNEL.shape) // 2))
        assert(np.all(origin2[:,1] == 10 + np.array(IDENTITY_KERNEL.shape) // 2))
        # assert that valid data is close
        np.testing.assert_allclose(out2[1+origin2[0,0]:1+origin2[0,1]-1, 1+origin2[1,0]:1+origin2[1,1]-1], input_data[11:20, 31:40], rtol=1e-5, atol=0)
    
    
    def test_fft_array_filter_check_data(self):
        """Test the fft_array_filter_check_data_method
        """
        # test 2d data
        nrow, ncol = 50, 60
        arr = np.arange(nrow*ncol, dtype=np.float32).reshape((nrow, ncol))
        
        fil_odd = np.zeros((4,4))
        fil_even, win, axes, conv_margins = fft_array_filter_check_data(
                arr, fil=fil_odd, win=None, zoom=1, axes=None)
        
        # check the filter shape is odd
        assert(np.all(fil_even.shape == (5,5)))
        # check the window is bidimensionnal
        assert(win.ndim == 2)
        # check the window shape matches (2,2)
        assert(np.all(win.shape == (2,2)))
        # check the window covers all data
        assert(np.all(win == ((0, nrow-1), (0, ncol-1))))
        # check axes are explicit and expected length
        assert(len(axes) == arr.ndim)
        # check the conv margins are ok
        assert(np.all(conv_margins == [fil_even.shape[0]//2,fil_even.shape[1]//2]))
    
    
    def test_fft_odd_filter(self):
        """Test the fft odd filter method"""
        # First check an already odd filter
        odd_filter = np.ones((5,5))
        ret_filter = fft_odd_filter(odd_filter, None)
        assert(np.all(odd_filter == ret_filter))
        # same check but only on axe 1
        ret_filter = fft_odd_filter(odd_filter, axes=(0,))
        assert(np.all(odd_filter == ret_filter))
        
        # Check a even filter
        even_filter = np.ones((4,4))
        ret_filter = fft_odd_filter(even_filter, None)
        assert(np.all(ret_filter.shape == (5,5)))
        assert(np.all(even_filter == ret_filter[0:4,0:4]))
        # check last column is 0
        assert(np.all(ret_filter[:,-1] == 0))
        # check last line is 0
        assert(np.all(ret_filter[-1,:] == 0))
        
        # same check but only on axe 1
        ret_filter = fft_odd_filter(even_filter, axes=(0,))
        assert(np.all(ret_filter.shape == (5,4)))
        assert(np.all(even_filter == ret_filter[0:4,0:4]))
        # check last line is 0
        assert(np.all(ret_filter[-1,:] == 0))
    
    
    def test_fft_filtering_output_shape(self):
        """Test the fft_filtering_output_shape method
        """
        nrow, ncol = 50, 60
        input_data = np.arange(nrow*ncol, dtype=np.float32).reshape((nrow, ncol))

        for boundary, out_mode in (
                (BoundaryPad.NONE, ConvolutionOutputMode.SAME),
                (BoundaryPad.REFLECT, ConvolutionOutputMode.SAME),
                (BoundaryPad.REFLECT, ConvolutionOutputMode.FULL),
                (BoundaryPad.NONE, ConvolutionOutputMode.FULL),
                (((BoundaryPad.NONE, BoundaryPad.NONE), (BoundaryPad.NONE, BoundaryPad.NONE)) , ConvolutionOutputMode.FULL),
                (((BoundaryPad.NONE, BoundaryPad.REFLECT), (BoundaryPad.NONE, BoundaryPad.NONE)) , ConvolutionOutputMode.FULL),
                ):
            shape_out = fft_array_filter_output_shape(
                arr=input_data,
                fil=IDENTITY_KERNEL,
                win=((10,20), (30,42)),
                boundary=boundary,
                out_mode=out_mode)
                
            out, _ = fft_array_filter(
                arr=input_data,
                fil=IDENTITY_KERNEL,
                win=((10,20), (30,42)),
                boundary=boundary,
                out_mode=out_mode,)
            
            if not np.all(shape_out == out.shape):
                raise Exception(f'test failed for boundary {boundary} out_mode {out_mode} : {shape_out}  VS {out.shape}')

        
        
        
        
        
