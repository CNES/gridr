# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of GRIDR
# (see https://gitlab.cnes.fr/gridr/gridr).
#
#
"""
Module for operations on array's window : check, extend, overflow
"""
import numpy as np

# Inside to outside signs for each edge
WINDOW_EDGE_OUTER_SIGNS = np.array((-1, 1))
# Outside to inside signs for each edge
WINDOW_EDGE_INNER_SIGNS = np.array((1, -1))

def window_check(arr: np.ndarray, win: np.ndarray, axes=None) -> bool:
    """Check a window lies inside an array shape.
    The method may raise an Exception if the order between index is not
    respected.
    
    Examples :
        arr = [[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12],
               [13, 14, 15, 16],
               [17, 18, 19, 20]]
        win = [[0,2], [3,3]]
    
    Args:
        arr: the array
        win: the window to test. The window is given as a list of tuple
                containing the first and last index for each dimension.
                e.g. for a dimension 2 : 
                ((first_row, last_row), (first_col, last_col))
             its dimension is not the same as the dimension of the input array :
             the number of tuple (first, last) must be equal to the number of
             dimensions in the array.
             If one or multiple axes are not taken, the corresponding tuple(s)
             must be set in the window in order to comply the the previous must.
        axes: axes on which the check is performed
    
    Returns:
        True if the window lies inside the array. False otherwise.
    """
    win = np.asarray(win)
    ret = True
    
    if arr.ndim == 0 or win.ndim == 0:  # scalar inputs
        raise ValueError("at least one input array is a scalar")
    elif arr.ndim != win.shape[0]:
        raise ValueError("array's number of dimension should be equal to the "
                "window's first dimension length")
    elif arr.size == 0 or win.size == 0:  # empty arrays
        ret = False
    
    if ret:
        if axes is None:
            axes = range(arr.ndim)
        axes = np.atleast_1d(axes)  # pylint: disable=R0204
        
        # check that first index is greater or equal the last index
        order_test = [np.nan if i not in axes else
                win[i][1]-win[i][0]>=0 for i in axes]
        # please note here that nan number are considered as True in np.all
        if ~np.all(order_test):
            raise Exception("At least one window's dimension range has invalid "
                    "order")
        
        # the order is ok ; now check that the window lies in the array
        within_test = [np.nan if i not in axes else
                win[i][0] >=0 and win[i][1] < arr.shape[i] for i in range(arr.ndim)]
        ret = np.all(within_test)  # pylint: disable=R0204
    return ret

def window_extend(
        win: np.ndarray,
        extent: np.ndarray,
        reverse: bool = False
        ) -> np.ndarray:
    """Extend a window.
    
    Args:
        win: the window to extend. The window is given as a list of tuple
                containing the first and last index for each dimension.
                e.g. for a dimension 2 : 
                ((first_row, last_row), (first_col, last_col))
        extent: the integer extents given as as a list of tuple
                containing the extent at boundaries for each dimension.
                e.g. for a dimension 2 : 
                ((up_extent, bottom_extent), (left_extent, right_extent))
        reverse: if false the extent is performed from inside to outside, if
                true it is performed from outside to inside.
    
    Returns:
        the extended window
    """
    win = np.asarray(win)
    extent = np.asarray(extent)
    
    signs = WINDOW_EDGE_OUTER_SIGNS
    if reverse:
        signs = WINDOW_EDGE_INNER_SIGNS
    return win + signs * extent

def window_overflow(arr: np.ndarray, win: np.ndarray, axes=None) -> np.ndarray:
    """Compute the overflow, i.e. the width on each side that goes outside the
    array shape.
    
    Please note that overflow are set to 0 on not selected axes.
    
    Args:
        arr: the array
        win: The window is given as a list of tuple
                containing the first and last index for each dimension.
                e.g. for a dimension 2 : 
                ((first_row, last_row), (first_col, last_col))
             its dimension is not the same as the dimension of the input array :
             the number of tuple (first, last) must be equal to the number of
             dimensions in the array.
             If one or multiple axes are not taken, the corresponding tuple(s)
             must be set in the window in order to comply the the previous must.
        axes: axes on which to compute the overflow
    
    Returns:
        the computed overflow array given as (top overflow, bottom overflow,
                left overflow, right_overflow)
    """
    win = np.asarray(win)
        
    if axes is None:
        axes = range(arr.ndim)
    axes = np.atleast_1d(axes)  # pylint: disable=R0204
    
    overflow = [[0, 0] if i not in axes else
            [abs(min(0, win[i][0])), max(0, win[i][1] - arr.shape[i]+1)]
            for i in range(arr.ndim)]

    return np.asarray(overflow)

