import os
import numpy as np

from gridr.core.utils.array_window import window_check, window_extend, window_overflow

class TestArrayWindow:
    
    def test_window_check(self):
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
    
    def test_window_extent(self):
        np.testing.assert_equal(window_extend(win=[(0,30), (0,60)], extent=[[1,2],[3,4]]), [(-1,32),(-3,64)])
        np.testing.assert_equal(window_extend(win=[(0,30), (0,60)], extent=[[1,2],[3,4]], reverse=True), [(1,28),(3,56)])
    
    def test_window_overflow(self):
        # test bidimensional array
        nrow, ncol = 4, 7
        data2d = np.arange(nrow*ncol, dtype=np.float32).reshape((nrow, ncol))
        
        np.testing.assert_equal(window_overflow(arr=data2d, win=[(0,3), (0,6)], axes=None), [(0,0), (0,0)])
        np.testing.assert_equal(window_overflow(arr=data2d, win=[(0,4), (0,6)], axes=None), [(0,1), (0,0)])
        np.testing.assert_equal(window_overflow(arr=data2d, win=[(0,4), (-4,15)], axes=None), [(0,1), (4,9)])
        np.testing.assert_equal(window_overflow(arr=data2d, win=[(0,4), (-4,15)], axes=0), [[0,1], [0, 0]])