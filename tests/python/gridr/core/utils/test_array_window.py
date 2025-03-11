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
import os
import numpy as np
import pytest
import rasterio

from gridr.core.utils.array_window import (
        window_indices, window_apply, window_check, window_extend,
        window_overflow, window_shape, as_rio_window)

ARRAY_00 = np.arange(4*7).reshape(4,7)
WIN_00_01, AXES_00_01, WIN_ARRAY_00_01, NOCHECK_EXPECT_00_01 = [(0,3), (0,6)], None, ARRAY_00[0:4, 0:7], ARRAY_00[0:4, 0:7]
WIN_00_02, AXES_00_02, WIN_ARRAY_00_02, NOCHECK_EXPECT_00_02 = [(0,4), (0,6)], None, ValueError, ARRAY_00[0:5, 0:7]
WIN_00_03, AXES_00_03, WIN_ARRAY_00_03, NOCHECK_EXPECT_00_03 = [(0,3), (0,7)], None, ValueError, ARRAY_00[0:4, 0:8]
WIN_00_04, AXES_00_04, WIN_ARRAY_00_04, NOCHECK_EXPECT_00_04 = [(1,2), (0,6)], None, ARRAY_00[1:3, 0:7], ARRAY_00[1:3, 0:7]
WIN_00_05, AXES_00_05, WIN_ARRAY_00_05, NOCHECK_EXPECT_00_05 = [(0,3), (3,4)], None, ARRAY_00[0:4, 3:5], ARRAY_00[0:4, 3:5]
WIN_00_06, AXES_00_06, WIN_ARRAY_00_06, NOCHECK_EXPECT_00_06 = [(1,2), (3,3)], None, ARRAY_00[1:3, 3:4], ARRAY_00[1:3, 3:4]
WIN_00_07, AXES_00_07, WIN_ARRAY_00_07, NOCHECK_EXPECT_00_07 = [(1,2), (3,3)], (0,), ARRAY_00[1:3, 0:7], ARRAY_00[1:3, 0:7]

class TestArrayWindow:
    """Class for test
    """
    @pytest.mark.parametrize("data, expected", [
            (([(2,3), (3,6)], False, None), (slice(2,4), slice(3,7))),
            (([(2,3), (3,6)], True, None), (slice(0,2), slice(0,4))),
            (([(2,3), (3,6)], False, [0]), (slice(2,4), slice(None,None))),
            (([(2,3), (3,6)], False, [1]), (slice(None,None), slice(3,7))),
            ])
    def test_window_indices(self, data, expected):
        """Test window_apply method
        """
        win, reset_origin, axes= data
        try:
            indices = window_indices(win, reset_origin, axes)
        except Exception as e:
            if isinstance(e, expected):
                pass
            else:
                raise
        else:
            try:
                if issubclass(expected, BaseException):
                    raise Exception(f"The test should have raised an exceptionof type {expected}")
            except:
                pass
            
            assert(indices == expected)
    
    
    @pytest.mark.parametrize("data, expected", [
            (([(2,3), (3,6)], None), (2,4)),
            (([(2,3), (3,6)], [0]), (2,None)),
            (([(2,3), (3,6)], [1]), (None,4)),
            ])
    def test_window_shape(self, data, expected):
        """Test window_apply method
        """
        win, axes= data
        try:
            shape = window_shape(win, axes)
        except Exception as e:
            if isinstance(e, expected):
                pass
            else:
                raise
        else:
            try:
                if issubclass(expected, BaseException):
                    raise Exception(f"The test should have raised an exceptionof type {expected}")
            except:
                pass
            
            assert(shape == expected)
    
    
    @pytest.mark.parametrize("data, expected", [
            (np.array([[2,3], [3,6]]), rasterio.windows.Window(3,2,4,2)),
            ])
    def test_as_rio_window(self, data, expected):
        """Test window_apply method
        """
        win = data
        try:
            win_rio = as_rio_window(win)
        except Exception as e:
            if isinstance(e, expected):
                pass
            else:
                raise
        else:
            try:
                if issubclass(expected, BaseException):
                    raise Exception(f"The test should have raised an exceptionof type {expected}")
            except:
                pass
            
            assert(win_rio == expected)
            
    
    @pytest.mark.parametrize("data, expected, testing_decimal", [
            ((ARRAY_00, WIN_00_01, AXES_00_01), (WIN_ARRAY_00_01, NOCHECK_EXPECT_00_01), 6),
            ((ARRAY_00, WIN_00_02, AXES_00_02), (WIN_ARRAY_00_02, NOCHECK_EXPECT_00_02), 6),
            ((ARRAY_00, WIN_00_03, AXES_00_03), (WIN_ARRAY_00_03, NOCHECK_EXPECT_00_03), 6),
            ((ARRAY_00, WIN_00_04, AXES_00_04), (WIN_ARRAY_00_04, NOCHECK_EXPECT_00_04), 6),
            ((ARRAY_00, WIN_00_05, AXES_00_05), (WIN_ARRAY_00_05, NOCHECK_EXPECT_00_05), 6),
            ((ARRAY_00, WIN_00_06, AXES_00_06), (WIN_ARRAY_00_06, NOCHECK_EXPECT_00_06), 6),
            ((ARRAY_00, WIN_00_07, AXES_00_07), (WIN_ARRAY_00_07, NOCHECK_EXPECT_00_07), 6),
            ])
    @pytest.mark.parametrize("check", [True, False])
    def test_window_apply(self, data, expected, check, testing_decimal):
        """Test window_apply method
        """
        array, win, axes = data
        expected_array, expected_nocheck = expected
        try:
            
            win_array = window_apply(array, win, axes, check=check)
        except Exception as e:
            if check:
                if isinstance(e, expected_array):
                    pass
            elif not check:
                if isinstance(e, expected_nocheck):
                    pass
            else:
                raise
        else:
            if check:
                try:
                    if issubclass(expected_array, BaseException):
                        raise Exception(f"The test should have raised an exceptionof type {expected_mask}")
                except TypeError:
                    pass
                
                try:
                    np.testing.assert_array_almost_equal(win_array, expected_array, decimal=testing_decimal)
                except TypeError:
                    raise Exception(f"Type error on {expected_array}")
            else:
                try:
                    if issubclass(expected_nocheck, BaseException):
                        raise Exception(f"The test should have raised an exceptionof type {expected_nocheck}. "
                                f"Instead it returns {win_array}")
                except TypeError:
                    pass
                
                try:
                    np.testing.assert_array_almost_equal(win_array, expected_nocheck, decimal=testing_decimal)
                except TypeError:
                    raise Exception(f"Type error on {expected_nocheck}")

    
    def test_window_check(self):
        """Test the window_check method
        """
        # test bidimensional array
        nrow, ncol = 4, 7
        data2d = np.arange(nrow*ncol, dtype=np.float32).reshape((nrow, ncol))
        
        # Test dimensions
        try:
            window_check(data2d, win=None, axes=None)
        except ValueError:
            # It is expected because win is considered as scalar.
            pass
        else:
            raise Exception("check should not have passed because win is scalar")
        
        try:
            window_check(data2d[0], win=[(0, 3), (0, 6)], axes=None)
        except ValueError:
            # It is expected because win has more elements (2) than the number
            # of dimension of data2d[0] (1)
            pass
        else:
            raise Exception("check should not have passed")
        
        try:
            window_check(data2d, win=[(0, 3)], axes=None)
        except ValueError:
            # It is expected because win has less elements (1) than the number
            # of dimension of data2d (2)
            pass
        else:
            raise Exception("check should not have passed")
        
        assert(window_check(data2d[0], win=[(0, 3)], axes=None))
        
        # Test window
        
        assert(window_check(data2d, win=[(0, 3), (0, 6)], axes=None))
        assert(window_check(data2d, win=[(1, 2), (3, 3)], axes=None))
        assert(~window_check(data2d, win=[(1, 2), (3, 7)], axes=None)) # overflow on axe 1
        assert(window_check(data2d, win=[(1, 2), (3, 7)], axes=0)) # check only on axe 0
        assert(window_check(data2d, win=[(1, 2), (3, 7)], axes=(0,))) # check only on axe 0
        assert(~window_check(data2d, win=[(1, 2), (3, 7)], axes=1)) # check only on axe 1
        
        # test empty arrays
        assert(~window_check(np.empty((0,0)), win=([0,0]), axes=None)) # empty array
        assert(~window_check(np.empty(1), win=[[]], axes=None)) # empty window
        
        # test window order
        try:
            assert(~window_check(data2d, win=[(0, 3), (6, 0)], axes=None))
        except Exception:
            # It is expected here because the second dimension order has been reversed
            pass
        else:
            raise Exception("window order check should not have passed")
        
    
    def test_window_extent(self):
        """Test the window_extent method
        """
        # Test outer extent
        np.testing.assert_equal(window_extend(win=[(0,30), (0,60)], extent=[[1,2],[3,4]]),
                [(-1,32),(-3,64)])
        # Test inner extent
        np.testing.assert_equal(window_extend(win=[(0,30), (0,60)], extent=[[1,2],[3,4]],
                reverse=True), [(1,28),(3,56)])
    
    def test_window_overflow(self):
        """Test the window_overflow method
        """
        # test a bidimensional array
        nrow, ncol = 4, 7
        data2d = np.arange(nrow*ncol, dtype=np.float32).reshape((nrow, ncol))
        # test case : window covers all data
        np.testing.assert_equal(window_overflow(arr=data2d, win=[(0,3), (0,6)], axes=None),
                [(0,0), (0,0)])
        # test case : window 1st dimension is greater (right) than the data 1st dimension
        np.testing.assert_equal(window_overflow(arr=data2d, win=[(0,4), (0,6)], axes=None),
                [(0,1), (0,0)])
        # test case :
        #    - window 1st dimension is greater (right) than the data 1st dimension and
        #    - window 2nd dimension is greater (left and right) than the data 2nd dimension
        #    - test performed on all axes
        np.testing.assert_equal(window_overflow(arr=data2d, win=[(0,4), (-4,15)], axes=None),
                [(0,1), (4,9)])
        # test case :
        #    - window 1st dimension is greater (right) than the data 1st dimension and
        #    - window 2nd dimension is greater (left and right) than the data 2nd dimension
        #    - test performed on 1st axe only => expect 0 overflow on second axe.
        np.testing.assert_equal(window_overflow(arr=data2d, win=[(0,4), (-4,15)], axes=0),
                [[0,1], [0, 0]])