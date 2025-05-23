# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of GRIDR
# (see https://gitlab.cnes.fr/gridr/gridr).
#
#
"""
Grid commons module
"""
from typing import List, Optional, Tuple, Union

import numpy as np

from gridr.core.utils.array_window import window_apply

def grid_full_resolution_shape(
        shape: Tuple[int, int],
        resolution: Tuple[int, int],
        ) -> Tuple[int, int]:
    """Compute the grid's shape at full resolution
    
    Args:
        shape: grid output shape as a tuple of integer (number of rows, number 
                of columns).
        resolution: grid resolution as a tuple of integer (row resolution,
                columns resolution)
    
    Returns:
        The grid's shape at full resolution
    """
    nrow = (shape[0]-1) * resolution[0] + 1
    ncol = (shape[1]-1) * resolution[1] + 1
    return (nrow, ncol)


def grid_regular_coords_1d(
        shape: Tuple[int, int],
        origin: Tuple[float, float],
        resolution: Tuple[int, int],
        ) -> Tuple[np.ndarray, np.ndarray]:
    """Create grid one-dimensionnal coordinates its output shape, origin and
    resolution.
    
    This method returns the columns and row coordinates array as 2 numpy
    ndarrays.
    
    The first array corresponds to the columns coordinates :
        G_col[j] = origin[1] + j * step_col 
    with
        step_col = resolution[1] * 1/ (shape[1]-1)
        
    The second array corresponds to the rows coordinates :
        G_row[i] = origin[0] + i * step_row
    with
        step_row = resolution[0] * 1/ (shape[0]-1)
    
    Args:
        shape: grid output shape as a tuple of integer (number of rows, number 
                of columns).
        origin: grid origin as a tuple of float (origin's row coordinate, 
                origin's column coordinate)
        resolution: grid resolution as a tuple of integer (row resolution,
                columns resolution)
    """
    x = np.linspace(origin[1], origin[1]+resolution[1]*(shape[1]-1), shape[1])
    y = np.linspace(origin[0], origin[0]+resolution[0]*(shape[0]-1), shape[0])
    return x, y


def grid_regular_coords_2d(
        shape: Tuple[int, int],
        origin: Tuple[float, float],
        resolution: Tuple[int, int],
        sparse: bool = False
        ) -> Tuple[np.ndarray, np.ndarray]:
    """Create 2D grid's coordinates considering its output shape, origin and
    resolution.
    
    This method returns the columns and row coordinates array as 2 numpy
    ndarrays.
    
    The first array corresponds to the columns coordinates :
        G_col[j,i] = origin[1] + j * step_col 
    with
        step_col = resolution[1] * 1/ (shape[1]-1)
        
    The second array corresponds to the rows coordinates :
        G_row[j,i] = origin[0] + i * step_row
    with
        step_row = resolution[0] * 1/ (shape[0]-1)
    
    The method use the numpy meshgrid function to create the grid. The sparse
    argument is directly passed to this function. Please see numpy.meshgrid for
    details about that argument.
    
    Args:
        shape: grid output shape as a tuple of integer (number of rows, number 
                of columns).
        origin: grid origin as a tuple of float (origin's row coordinate, 
                origin's column coordinate)
        resolution: grid resolution as a tuple of integer (row resolution,
                columns resolution)
    """
    x, y = grid_regular_coords_1d(shape, origin, resolution)
    xx, yy = np.meshgrid(x, y, indexing='xy', sparse=sparse)
    return xx, yy


def regular_grid_shape_origin_resolution(
        grid_coords: Union[Tuple[np.ndarray], List[np.ndarray], np.ndarray]
        ) -> Tuple[Tuple[int, int], Tuple[float, float], Tuple[int, int]]:
    """Compute shape, origin and resolution from a regular grid.
    
    The grid can be given either as a 3D grid, a tuple of 2 2D arrays or a tuple
    of 2 1D arrays.
    
    Args:
        grid_coords: grid corresponding coordinates. If not given they are
                The grid coordinates are given as a 3d arrays or a tuple of 1d
                or 2d arrays containing the columns and the rows of pixels
                centroïds.

    Returns:
        shape, origin, resolution
    """
    shape, origin, resolution = None, None, None
    try:
        if grid_coords.ndim == 3:
            shape = grid_coords[0].shape
            origin = (grid_coords[1, 0, 0], grid_coords[0, 0, 0])
            resolution = (
                    (grid_coords[1,-1,0] - grid_coords[1,0,0]) / (shape[0] - 1),
                    (grid_coords[0,0,-1] - grid_coords[0,0,0]) / (shape[1] - 1))
    except AttributeError:
        if grid_coords[0].ndim == 2:            
            shape = grid_coords[0].shape
            origin = (grid_coords[1][0, 0], grid_coords[0][0, 0])
            resolution = (
                    (grid_coords[1][-1,0] - grid_coords[1][0,0]) / (shape[0] - 1),
                    (grid_coords[0][0,-1] - grid_coords[0][0,0]) / (shape[1] - 1))
        else:
            shape = (len(grid_coords[1]), len(grid_coords[0]))
            origin = (grid_coords[1][0], grid_coords[0][0])
            resolution = (
                    (grid_coords[1][-1] - grid_coords[1][0]) / (shape[0] - 1),
                    (grid_coords[0][-1] - grid_coords[0][0]) / (shape[1] - 1))
            
    return shape, origin, resolution


def window_apply_grid_coords(
        grid_coords: Union[Tuple[np.ndarray], List[np.ndarray], np.ndarray],
        win: np.ndarray,
        check: bool = True,
        ) -> Tuple[np.ndarray]:
    """Apply a window to a grid.
    
    The grid can be given either as a 3D grid, a tuple of 2 2D arrays or a tuple
    of 2 1D arrays.
    
    Args:
        grid_coords: grid corresponding coordinates. If not given they are
                The grid coordinates are given as a 3d arrays or a tuple of 1d
                or 2d arrays containing the columns and the rows of pixels
                centroïds.
        win: the window to apply. The window is given as a list of tuple
                containing the first and last index for each dimension.
                e.g. for a dimension 2 : 
                ((first_row, last_row), (first_col, last_col))
             its dimension is not the same as the dimension of the input array :
             the number of tuple (first, last) must be equal to the number of
             dimensions in the array.
             If one or multiple axes are not taken, the corresponding tuple(s)
             must be set in the window in order to comply the the previous must.
        check: boolean option to activate window_apply input's consistency check
    
    Returns:
        The windowed grid as a tuple of 2D or 1D coordinates
    """
    windowed_grid = None
    try:
        if grid_coords.ndim == 3:
            windowed_grid = (window_apply(grid_coords[0], win=win, check=check),
                    window_apply(grid_coords[1], win=win, check=check))
    except AttributeError:
        if grid_coords[0].ndim == 2:
            windowed_grid = (window_apply(grid_coords[0], win=win, check=check),
                    window_apply(grid_coords[1], win=win, check=check))
        else:
            # grid coords are 1d
            # but win does respect a 2d view.
            windowed_grid = (window_apply(grid_coords[0], win=[win[1]], check=check),
                    window_apply(grid_coords[1], win=[win[0]], check=check))
                        
    return windowed_grid


def window_apply_shape_origin_resolution(
        shape: Tuple[int, int],
        origin: Tuple[float, float],
        resolution: Tuple[int, int],
        win: np.ndarray,
        ) -> Tuple[Tuple[int, int], Tuple[float, float], Tuple[int, int]]:
    """Apply a window to the 3 arguments shape, origin and resolution
    
    Args:
        shape: grid output shape as a tuple of integer (number of rows, number 
                of columns).
        origin: grid origin as a tuple of float (origin's row coordinate, 
                origin's column coordinate)
        resolution: grid resolution as a tuple of integer (row resolution,
                columns resolution)
        win: the window to apply. The window is given as a list of tuple
                containing the first and last index for each dimension.
                e.g. for a dimension 2 : 
                ((first_row, last_row), (first_col, last_col))
             its dimension is not the same as the dimension of the input array :
             the number of tuple (first, last) must be equal to the number of
             dimensions in the array.
             If one or multiple axes are not taken, the corresponding tuple(s)
             must be set in the window in order to comply the the previous must.

    Returns:
        shape, origin, resolution
    """
    shape_out, origin_out, resolution_out \
            = [None, None], [None, None], [None, None]
    
    assert(np.all(win.shape == (2,2)))
    origin_out[0] = origin[0] + win[0][0] * resolution[0]
    origin_out[1] = origin[1] + win[1][0] * resolution[1]
    
    shape_out[0] = win[0][1] - win[0][0] + 1
    shape_out[1] = win[1][1] - win[1][0] + 1
    
    resolution_out = resolution
    
    return shape_out, origin_out, resolution_out


def check_grid_coords_definition(
        grid_coords: Optional[Tuple[np.ndarray, np.ndarray]],
        shape: Optional[Tuple[int, int]],
        origin: Optional[Tuple[float, float]],
        resolution: Optional[Tuple[int, int]],
        ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Check grid definition's parameters.
    
    Args:
        grid_coords: grid corresponding coordinates. If not given they are
                computed with shape, origin and resolution arguments.
                The grid coordinates are given as tuple of 1d or 2d arrays 
                containing the columns and the rows of pixels centroïds given
                in the same frame as the geometry.
        shape: grid output shape as a tuple of integer (number of rows, number 
                of columns).
        origin: grid origin as a tuple of float (origin's row coordinate, 
                origin's column coordinate)
        resolution: grid resolution as a tuple of integer (row resolution,
                columns resolution)
        
    Returns:
        Updated grid_coords (if needed)
    """
    # Check computation domain arguments
    # We make sure here that the coordinates are either given through the
    # grid_coords argument or the 3 arguments 'shape', 'origin' and 'resolution'
    if grid_coords is not None \
            and shape is None and origin is None and resolution is None:
        try:
            if len(grid_coords) != 2:
                raise TypeError("The grid_coords argument first dimension must "
                        "contain 2 elements for columns and grids coordinates")
            
            if not isinstance(grid_coords[0], np.ndarray) \
                    or not isinstance(grid_coords[1], np.ndarray):
                raise TypeError("The grid_coords argument items must be numpy "
                        "ndarrays")
            
            if grid_coords[0].ndim != grid_coords[1].ndim:
                raise TypeError("The grid_coords items must have the same "
                        " dimension")
            
            if grid_coords[0].ndim not in (1, 2):
                raise TypeError("The grid_coords items must be of dimension 1 "
                        "or 2")
        except IndexError as e:
            raise TypeError("The grid_coords argument should be indexable and "
                    "have at least 2 dimensions.") from e
        
        # Check grid_coords input type and adapt it to be a tuple of 2 2d arrays
        try:
            if grid_coords.ndim == 3:
                grid_coords = (grid_coords[0], grid_coords[1])
            
        except AttributeError:
            # We consider grid_coords is already an iterable
            pass
        
    elif grid_coords is None \
            and shape is not None and origin is not None \
            and resolution is not None:
        # TODO : more checks here
        pass
        
    else:
        raise ValueError("You should either provide the grid_coords argument or the " 
                "3 arguments 'shape', 'origin' and 'resolution'")
    
    return grid_coords


def grid_resolution_window(
        resolution: Tuple[int, int],
        win: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    This method computes the required window needed in the input grid in order
    to cover a full target resolution given through the 'window' parameter.
    If grid's oversampling are equal to 1 in both direction, the required window
    is directly equal to the input window.
    
    Args:
        resolution : The grid's resolution factors for rows and columns
        win: The target full resolution window
    """
    resolution_arr = np.asarray(resolution)
    
    # Compute the index in grid
    # - start index : we must take the nearest lower index in the input grid. It
    #        is given by the integer division of the target index by the 
    #        resolution along each axis.
    # - stop index : we must take the nearest upper index in the input grid.
    grid_win = np.vstack((win[:, 0] // resolution_arr,
            win[:, 1] // resolution_arr \
            + (win[:,1] % resolution_arr != 0).astype(int))).T
            
    # Compute the relative position in the grid_win in order to target the 
    # target win
    rel_win = np.vstack((win[:,0] % resolution_arr,
            win[:,0] % resolution_arr + win[:,1] - win[:,0])).T
    
    return grid_win, rel_win
    

def grid_resolution_window_safe(
        resolution: Tuple[int, int],
        win: np.ndarray,
        grid_shape : Tuple[int, int],
        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    This method computes the required window needed in the input grid in order
    to cover a full target resolution given through the 'window' parameter.
    If grid's oversampling are equal to 1 in both direction, the required window
    is directly equal to the input window.
    
    Args:
        resolution : The grid's resolution factors for rows and columns
        win: The target full resolution window
        grid_shape: The total shape (rows, columns) of the input grid.
    """
    resolution_arr = np.asarray(resolution)
    grid_shape_arr = np.asarray(grid_shape)
    
    # Compute the index in grid
    # - start index : we must take the nearest lower index in the input grid. It
    #        is given by the integer division of the target index by the 
    #        resolution along each axis.
    # - stop index : we must take the nearest upper index in the input grid.
    grid_win_start = win[:, 0] // resolution_arr
    grid_win_stop = win[:, 1] // resolution_arr \
            + (win[:,1] % resolution_arr != 0).astype(int)
    
    # Store original start for rel_win adjustment
    original_grid_win_start = np.copy(grid_win_start)

    # For each dimension where resolution > 1, ensure stop - start + 1 >= 2
    mask_resolution_gt_1 = resolution_arr > 1
    current_size = grid_win_stop - grid_win_start
    needs_extension_mask = (current_size < 1) & mask_resolution_gt_1
    
    # Extend stop if needed, but not beyond grid_shape_arr - 1
    # This addresses cases like [5,5] needing to become [5,6] but grid max is 5
    grid_win_stop[needs_extension_mask] = np.minimum(
            grid_win_stop[needs_extension_mask] + 1,
            grid_shape_arr[needs_extension_mask] - 1)
    
    # Re-evaluate current_size after initial stop extension
    current_size = grid_win_stop - grid_win_start
    
    # Identify where size is still less than 1 (meaning stop is still <= start)
    # This handles cases where original stop was already at boundary 
    # (e.g., [5,5] and grid_shape is 6) and where resolution_arr > 1
    needs_extension_mask = (current_size < 1) & mask_resolution_gt_1

    # If still too small, try to shift start backward, but not below 0
    # This handles cases like [5,5] becoming [5,5] because grid max was 5, now
    # try [4,5]
    grid_win_start[needs_extension_mask] = np.maximum(
            grid_win_start[needs_extension_mask] - 1, 0)
    
    # Final check: ensure stop is not less than start for validity, but don't
    # force minimum size here (The above logic already ensures minimum size
    # where possible)
    # This primarily addresses cases where start was shifted back and might have 
    # become larger than stop
    # This shouldn't happen with the logic above if grid_shape allows a 2-pixel
    # window, but it's a safeguard for extreme edge cases where grid_shape
    # itself is too small.
    # For example, if grid_shape is 1, it's impossible to have a 2-pixel window.
    grid_win_stop = np.maximum(grid_win_stop, grid_win_start)

    # Ensure grid_win_start is not negative and grid_win_stop is within bounds
    # (redundant with previous checks but good for final clamp)
    grid_win_start = np.maximum(grid_win_start, 0)
    grid_win_stop = np.minimum(grid_win_stop, grid_shape_arr - 1)
    
    
    grid_win = np.vstack((grid_win_start, grid_win_stop)).T

    # Compute the relative position in the grid_win in order to target the
    # target win
    # We need to adjust rel_win_start based on the *actual* grid_win_start
    # compared to the original, ideal grid_win_start.
    rel_win_start_adjusted = win[:,0] % resolution_arr + \
            (original_grid_win_start - grid_win_start) * resolution_arr
    
    # The relative stop should just be the relative start plus the span of the 
    # target window
    rel_win_stop = rel_win_start_adjusted + (win[:,1] - win[:,0])
    
    # Max possible relative coordinate within the read grid_win, considering
    # resolution.
    # (grid_win_stop - grid_win_start) is the number of cells in the 
    # grid_win minus 1
    # Multiplied by resolution_arr and adding 1, this gives the number of
    # elements in "high-resolution"
    # For the max index we have to substract 1, leading to :
    max_rel_coord_index = (grid_win_stop - grid_win_start) * resolution_arr
    
    rel_win_stop = np.minimum(rel_win_stop, max_rel_coord_index)

    # Ensure rel_win_start is not negative (should be handled by modulo and 
    # shift logic but good safeguard)
    rel_win_start = np.maximum(rel_win_start_adjusted, 0)
    
    # Ensure rel_win_stop is not less than rel_win_start, creating a valid 
    # relative window
    rel_win_stop = np.maximum(rel_win_stop, rel_win_start)

    rel_win = np.vstack((rel_win_start, rel_win_stop)).T

    return grid_win, rel_win