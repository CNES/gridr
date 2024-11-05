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
"""
import os
import numpy as np

from gridr.core.utils.array_window import (
        window_check, window_extend, window_overflow)

class TestArrayWindow:
    """Class for test
    """
    
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