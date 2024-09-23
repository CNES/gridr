import numpy as np

# Inside to outside signs for each edge (right, left, top, bottom) of a window
WINDOW_EDGE_OUTER_SIGNS = np.array((-1, 1, -1, 1))
# Outside to inside signs for each edge (right, left, top, bottom) of a window
WINDOW_EDGE_INNER_SIGNS = np.array((1, -1, 1, -1))

def window_check(arr: np.ndarray, win: np.ndarray):
    """Check a window lies inside an array shape.
    The method may raise an Exception if the order between index is not
    respected.
    
    Args:
        arr: the array
        win: the window to test. The window is given as (index_first_row,
                index_last_row, index_first_col, index_last_col)
    
    Returns:
        True if the window lies inside the array. False otherwise.
    """
    if window[0] > window[1]:
        raise Exception("Window first row index is greater than last row index")
    if window[2] > window[3]:
        raise Exception("Window first column index is greater than last column index")

    # The first test check that last row is greater han first row
    # We do not have to test that window[1] < 0 and window[0] >= arr.shape[0]
    check_row = window[0] >= 0 and window[1] < arr.shape[0]
    # same for column
    check_col = window[2] >= 0 and window[3] < arr.shape[1]
    
    return check_row and check_col

def window_extend(
        window: np.ndarray,
        extent: np.ndarray,
        reverse: bool = False
        ) -> np.ndarray:
    """Extend a window.
    
    Args:
        win: the window to extend. The window is given as (index_first_row,
                index_last_row, index_first_col, index_last_col)
        extent: the integer extents given as (top_extent, bottom_extent, 
                left_extent, right_extent)
        reverse: if false the extent is performed from inside to outside, if
                true it is performed from outside to inside.
    
    Returns:
        the extended window
    """
    signs = WINDOW_EDGE_OUTER_SIGNS
    if reverse:
        signs = WINDOW_EDGE_INNER_SIGNS
    return window + signs * extent

def window_overflow(arr: np.ndarray, win: np.ndarray):
    """Compute the overflow, i.e. the width on each side that goes outside the
    array shape.
    
    Args:
        arr: the array
        win: the window given as (index_first_row, index_last_row,
                index_first_col, index_last_col)
    
    Returns:
        the computed overflow array given as (top overflow, bottom overflow,
                left overflow, right_overflow)
    """
    overflow = np.array((abs(min(0, win[0])), # top
            max(0, win[1] - arr.shape[0] + 1), # bottom
            abs(min(0, win[2])), # left
            max(0, win[3] - arr.shape[1] + 1))) # right
    return overflow