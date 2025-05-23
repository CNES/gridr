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
from typing import List, Tuple, Optional

import numpy as np
from rasterio.windows import Window

# Inside to outside signs for each edge
WINDOW_EDGE_OUTER_SIGNS = np.array((-1, 1))
# Outside to inside signs for each edge
WINDOW_EDGE_INNER_SIGNS = np.array((1, -1))

def window_expand_ndim(
        win: np.ndarray,
        insert: np.ndarray,
        pos: int = 0) -> np.ndarray:
    """Expand a window by inserting at begin or end a new dimension.
    
    Args:
        win : the input window (it will not be modified)
        insert : the element to insert
        pos : the position of insertion : 0 at begin, -1 at the end.
    
    Returns:
        the expanded window
    """
    if pos not in (0, -1):
        raise ValueError("The argument 'pos' must be either 0 or -1")
    insert = np.copy(np.asarray(insert))
    win = np.copy(win)
    
    if pos == 0:
        win = np.vstack((insert, win[:]))
    else:
        win = np.vstack((win[:], insert))
    return win

def window_shift(
        win: np.ndarray,
        shift: np.ndarray,
        ) -> np.ndarray:
    """Shift an existing window from a scalar bias defined for each dimension
    
    e.g. :
    [(a,b), (c,d)] => [(a+u,b+u), (c+v, d+v)]
    
    Args:
        win: The window is given as a list of tuple
                containing the first and last index for each dimension.
                e.g. for a dimension 2 : 
                ((first_row, last_row), (first_col, last_col))
             its dimension is not the same as the dimension of the input array :
             the number of tuple (first, last) must be equal to the number of
             dimensions in the array.
             If one or multiple axes are not taken, the corresponding tuple(s)
             must be set in the window in order to comply the the previous must.
        shift: The shift given as an array containing shift for each dimension
    
    Returns:
        the shifted window
    """
    assert(shift.ndim == 1)
    assert(shift.shape[0] == win.shape[-2])
    return np.swapaxes(np.swapaxes(win,-2,-1) + shift,-2,-1)

def window_from_chunk(
        chunk: np.ndarray,
        origin: Optional[np.ndarray] = None,
        ) -> np.ndarray:
    """Returns a window from a chunk definition.
    
    The chunk is defined as a list of (start, stop) indices to be used 
    directly as slice when adressing a python array. Therefore it does not use
    the same convention as a window.
    The 'stop' index in the chunk will not be part of the returned view, whereas
    the 'last_<dim>' value for the window is an element that is part of the 
    returned view.
    
    Args:
        chunk: n-dimensionnal chunk definitions
        origin: 1-dimensionnal array containing bias to apply on adressed 
                dimension. The ith element in origin array is applied on the
                ith axes. This argument is optional.
    
    Returns:
        returns the chunk in the window's convention.
    """
    win = np.asarray(chunk)
    # change convention
    win[...,1] -= 1
    if origin is not None:
        win = window_shift(win, origin)
    return win

def window_indices(
        win: np.ndarray,
        reset_origin: bool = False,
        axes=None
        ) -> Tuple[slice]:
    """Get indices to use with an array to get the corresponding view using
    slices.

    Args:
        win: The window is given as a list of tuple
                containing the first and last index for each dimension.
                e.g. for a dimension 2 : 
                ((first_row, last_row), (first_col, last_col))
             its dimension is not the same as the dimension of the input array :
             the number of tuple (first, last) must be equal to the number of
             dimensions in the array.
             If one or multiple axes are not taken, the corresponding tuple(s)
             must be set in the window in order to comply the the previous must.
        reset_origin: boolean to substract each window interval of its first
             element. Thus resulting in slices whose first elements are 0.
        axes: axes on which to consider the window
    
    """
    win = np.asarray(win)
    
    if axes is None:
        axes = range(win.shape[0])
        axes = np.atleast_1d(axes)  # pylint: disable=R0204
    
    if reset_origin:
        win = win - np.vstack(win[:,0])
    
    indices = tuple((slice(None, None) if i not in axes else
            slice(int(win[i][0]),int(win[i][1]+1))
            for i in range(win.shape[0])))
    return indices

def window_from_indices(
    indices: Tuple[slice],
    original_shape: Tuple[int, ...],
    axes: Optional[List[int]] = None,
) -> np.ndarray:
    """Get the window (win) from the indices (slices).

    Args:
        indices: A tuple of slice objects, as returned by window_indices.
        original_shape: The shape on which the indices/win will be applied.
        axes: The axes that were considered when creating the indices.
              If None, it's assumed all axes were considered.

    Returns:
        win: The reconstructed window as a numpy array from a tuple of slices.
    """
    ndim = len(indices)
    win = np.zeros((ndim, 2), dtype=int)

    if axes is None:
        axes = range(ndim)
    axes = np.atleast_1d(axes)

    for i in range(ndim):
        if i in axes:
            s = indices[i]
            # Invert slice(start, stop) to get (start, stop-1)
            # The stop value in slice is exclusive, so we subtract 1 to get the 
            # last index
            start = s.start if s.start is not None else 0
            # Default to last element if None
            stop = s.stop - 1 if s.stop is not None else original_shape[i] - 1 

            win[i, 0] = start
            win[i, 1] = stop
        else:
            # For axes not included in 'axes' (which result in slice(None, None)),
            # the window effectively covers the entire dimension of the original
            # array.
            # So, we set the start to 0 and the end to original_shape[i] - 1.
            win[i, 0] = 0
            win[i, 1] = original_shape[i] - 1

    return win

def window_apply(
        arr: np.ndarray,
        win: np.ndarray,
        axes=None,
        check=True
        ) -> np.ndarray:
    """Apply the window to the array and return the windowed view
    
    User can disable consistency check between the array and the window. This
    option is provided in order to not perform a check if it has already been
    performed.
    Please note that if check is disabled, the check that the window lies inside
    the array is not performed. Numpy does not raise an IndexError exception in 
    such case ; it simply limits the window to the available data thus resulting
    in awkward behaviour.
    
    Args:
        arr: the array
        win: the window to apply. The window is given as a list of tuple
                containing the first and last index for each dimension.
                e.g. for a dimension 2 : 
                ((first_row, last_row), (first_col, last_col))
             its dimension is not the same as the dimension of the input array :
             the number of tuple (first, last) must be equal to the number of
             dimensions in the array.
             If one or multiple axes are not taken, the corresponding tuple(s)
             must be set in the window in order to comply the the previous must.
        axes: axes on which to apply the window
        check: boolean option to activate input's consistency checks
    
    Returns:
        The windowed array's view
    """
    win = np.asarray(win)
    ret = arr
    
    if check:
        if not window_check(arr, win, axes):
            raise ValueError("window check fails : check window/array "
                    "consistency")
    indices = window_indices(win, reset_origin=False, axes=axes)
    ret = arr[indices]
    return ret
    

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
            raise IndexError("At least one window's dimension range has invalid "
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

def window_shape(win: np.ndarray, axes=None) -> Tuple:
    """Compute the window's shape
    
    Args:
        win: The window is given as a list of tuple
                containing the first and last index for each dimension.
                e.g. for a dimension 2 : 
                ((first_row, last_row), (first_col, last_col))
             its dimension is not the same as the dimension of the input array :
             the number of tuple (first, last) must be equal to the number of
             dimensions in the array.
             If one or multiple axes are not taken, the corresponding tuple(s)
             must be set in the window in order to comply the the previous must.
        axes: axes on which to compute the length
    
    Returns:
        the window's shape
    """
    win = np.asarray(win)
        
    if axes is None:
        axes = range(win.shape[0])
    axes = np.atleast_1d(axes)  # pylint: disable=R0204
    
    shape = tuple([None if i not in axes else win[i,1] - win[i,0] + 1
            for i in range(win.shape[0])])
    return shape

def as_rio_window(win: np.ndarray) -> Window:
    """Convert to rasterio Window object.
    
    Args:
        win : The window is given as a list of tuple
                containing the first and last index for each dimension.
                e.g. for a dimension 2 : 
                ((first_row, last_row), (first_col, last_col))
             its dimension is not the same as the dimension of the input array :
             the number of tuple (first, last) must be equal to the number of
             dimensions in the array.
             If one or multiple axes are not taken, the corresponding tuple(s)
             must be set in the window in order to comply the the previous must.
    
    Returns:
        The corresponding rasterio.windows.Window object
    """
    win = np.asarray(win)
    args = [(incl_idx_0, incl_idx_1+1) for incl_idx_0, incl_idx_1 in win]
    return Window.from_slices(*args)

def from_rio_window(rio_win: Window) -> np.ndarray:
    """Convert a rasterio Window object to a GridR window
    
    Args:
        rio_win : The rasterio.windows.Window window
    
    Returns:
        The corresponding window
    """
    win = np.array([[rio_win.row_off, rio_win.row_off + rio_win.height -1],
            [rio_win.col_off, rio_win.col_off + rio_win.width -1]])
    return win
    
    