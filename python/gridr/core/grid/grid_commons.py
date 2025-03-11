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
    resolution = np.asarray(resolution)
    
    # Compute the index in grid
    # - start index : we must take the nearest lower index in the input grid. It
    #        is given by the integer division of the target index by the 
    #        resolution along each axis.
    # - stop index : we must take the nearest upper index in the input grid.
    grid_win = np.vstack((win[:, 0] // resolution,
            win[:, 1] // resolution \
            + (win[:,1] % resolution != 0).astype(int))).T
            
    # Compute the relative position in the grid_win in order to target the 
    # target win
    rel_win = np.vstack((win[:,0] % resolution,
            win[:,0] % resolution + win[:,1] - win[:,0])).T
    
    return grid_win, rel_win