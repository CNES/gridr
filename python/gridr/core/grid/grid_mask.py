# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of GRIDR
# (see https://gitlab.cnes.fr/gridr/gridr).
#
#
"""
Grid mask module
"""
from enum import IntEnum
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union, NoReturn, Optional

import numpy as np
import shapely

from gridr.core.utils.array_utils import ArrayProfile
from gridr.core.utils.array_window import (window_shape, window_check,
        window_apply)
from gridr.core.grid.grid_utils import oversample_regular_grid
from gridr.core.grid.grid_rasterize import (grid_rasterize,
        GridRasterizeAlg, ShapelyPredicate)


def build_mask(
        shape: Tuple[int, int],
        resolution: Tuple[int, int],
        out: np.ndarray,
        geometry_origin: Tuple[float, float],
        geometry: Optional[Union[shapely.geometry.Polygon,
                List[shapely.geometry.Polygon], shapely.geometry.MultiPolygon]],
        mask_in: Optional[np.ndarray],
        mask_in_target_win: np.ndarray,
        mask_in_resolution: Optional[Tuple[int, int]],
        oversampling_dtype: np.dtype,
        mask_in_binary_threshold: float = 1e-3,
        rasterize_kwargs: Optional[Dict] = None,
        ) -> Optional[np.ndarray]:
    """Create a binary mask that will be associated to a grid.
    
    This method only works on raster and do not perform IO.
    
    This method takes two different kind of masks and compile their information
    in order to build a binary raster mask at a target resolution (currently
    only the full resolution is implemented, i.e. (1,1)) :
    
    1) The first kind of mask, is a binary raster mask that can be given in a
    lower resolution through the 'mask_in' argument. This kind of mask usually
    comes with the grid at the same shape and resolution than the grid itself.
    Therefore the associated resolution become a mandatory argument if 'mask_in'
    is set.
    The 'mask_in_target_win' can also be given in order to only consider a 
    window of the resampled mask. It is given in the coordinates system of the
    resampled mask at output resolution.
    Given that the resampling of the mask may give floating point values, the
    'mask_in_binary_threshold' argument is used as a threshold to binarize the
    result of the mask's values interpolation. Any values greater or equal to 
    that argument are set to 1 (considered masked), 0 otherwise.
    
    2) The second kind of mask is given as vectors through the 'geometry'
    argument. It is given as a polygon or an union of polygons whose interior
    and contour (depending on the chosen algorithm's predicate) are considered
    unmasked. What is not will be considered masked.
    Such masks will be rasterized given the output coordinates system defined
    through the 'shape', 'resolution' and 'geometry_origin' arguments.
    
    A pixel will be set as masked (value 1) if it is either masked by the
    input raster mask or by the geometry vector mask.
    
    A preallocated output buffer can be passed to the method through the 'out'
    arugment. If it is given it must be consistent with the given 'shape'.
    
    Adopted conventions :
    - the masked pixels are set to 1, 0 otherwise.
    - the points used in geometries are given with (x, y) coordinates where
    x is the column et y the row.
    - all shape, resolution and geometry_origin are given as (value for row,
    value for column). Please note that the geometry_origin is not given
    in the same convention order than the point used in geometry definition.
    
    Args:
        shape: the shape of the output mask (optional). If not given it will be
                defined from the window (see win argument) or from the out
                buffer (see out argument) in that priority order.
        resolution: the resolution of the output mask. Only full resolution (ie
                (1,1) is currently implemented. The resolution is used for the
                resampling (oversampling) of the optional input mask and for the
                rasterization of the geometries.
        out: an optional preallocated buffer to store the result
        geometries_origin: geometric coordinates that are mapped to the output
                first pixel indexed by (0,0) in the array. This argument is
                mandatory if geometries is set.
        geometries: Definition of non masked geometries as a polygon or a list
                of polygons.
        mask_in: optional input raster mask
        mask_in_target_win: an optional production window given as a list of  
                tuple containing the first and last index for each dimension.
                e.g. for a dimension 2 : 
                ((first_row, last_row), (first_col, last_col))
        mask_in_resolution: resolution in row and column of input raster mask
        oversampling_dtype: the data type to use for oversampling the mask. It
                must be a floating type.
        mask_in_binary_threshold: in case of binary option is activated, all
                values greater or equal to mask_in_binary_threshold are set to
                1, 0 otherwise.
        rasterize_kwargs: dictionnary of parameters for the rasterize process.
                egg. {'alg': GridRasterizeAlg.SHAPELY,
                'kwargs_alg': {'shapely_predicate': ShapelyPredicate.COVERS}
    """
    ret = None
    # -- Perform some checks on arguments and init optional arguments
    if shape is None or resolution is None:
        raise ValueError("You must provide both the 'shape' and 'resolution' "
                "arguments")
    if resolution is None:
        raise ValueError("You must provide the 'resolution' argument")
    if ~np.all(resolution == (1,1)):
        raise ValueError("Output resolution different from full resolution have"
                " not been implemented yet")
    
    if geometry is not None and geometry_origin is None:
        raise ValueError("You must provide the 'geometry_origin' argument "
                "in order to use rasterization through the 'geometry' argument")
    
    if mask_in is not None and not mask_in_resolution:
        raise ValueError("You must provide the 'mask_in_resolution' argument")
        
    # Init output buffer if not given
    if out is None:
        out = np.zeros(shape, dtype=np.uint8)
        ret = out
    elif ~np.all(out.shape == shape):
        raise ValueError("The values of the 2 arguments 'out' and 'shape' does "
                "not match.")
    
    if oversampling_dtype is not None:
        if not np.issubdtype(oversampling_dtype, np.floating):
            raise ValueError("The value of argument 'oversampling_dtype' is not"
                    " a floating type")
    else:
        raise ValueError("You must precise argument 'oversampling_dtype'")
    
    # At last check that the window lies in the full_resolution input mask
    # if given
    if mask_in is not None:
        if mask_in_target_win is None:
            # Define a correct 2d window matching the array dimensions
            mask_in_target_win = [(0, shape[i]-1) for i in range(2)]
        elif ~np.all(shape == window_shape(mask_in_target_win)):
            raise ValueError(f"The shapes of the 2 arguments 'shape' ({shape}) "
                    f"and 'mask_in_target_win' "
                    f"({window_shape(mask_in_target_win)}) does not match.")
                    
        mask_in_target_win = np.asarray(mask_in_target_win)
        
        # Compute the mask in profile at full resolution
        # FUTURE_WARNING : if resolution_out != 1 we will have to update this
        # code
        mask_in_full_res_profile = ArrayProfile(
                shape=((mask_in.shape[0]-1)*mask_in_resolution[0]+1,
                        (mask_in.shape[1]-1)*mask_in_resolution[1]+1),
                ndim=mask_in.ndim, dtype=mask_in.dtype)
        if not window_check(arr=mask_in_full_res_profile,
                win=mask_in_target_win, axes=None):
            raise ValueError(
                    "Target window error is not contained in input mask : "
                    f"\n\t Input mask : {mask_in_full_res_profile.shape}"
                    f"\n\t Window : {mask_in_target_win}")
    # -- End of argument's checks
    
    # Compute the full resolution binary mask if given
    merge = False
    if mask_in is not None:
        merge = True
        if mask_in_resolution[0] != 1 or mask_in_resolution[1] != 1:
            # We have to oversample the mask to the output resolution
            # FUTURE_WARNING : if resolution_out != (1,1) we will have to
            # reimplement this part
            _, out[:,:] = oversample_regular_grid(
                    grid = None,
                    grid_oversampling_row=mask_in_resolution[0],
                    grid_oversampling_col=mask_in_resolution[1],
                    grid_mask=mask_in,
                    grid_mask_binarize_precision = mask_in_binary_threshold,
                    grid_mask_dtype = out.dtype,
                    win = mask_in_target_win,
                    dtype = oversampling_dtype, # dtype used for interpolation
                    )
        else:
            # No oversampling - apply the window selection and binarize it
            # FUTURE_WARNING : if resolution_out != (1,1) we will have to
            # reimplement this part
            windows_mask_in = window_apply(arr=mask_in, win=mask_in_target_win)
            out[:,:] = np.abs(windows_mask_in) >= mask_in_binary_threshold
    
    # Rasterize the geometry if it has been set
    if geometry is not None:
        grid_rasterize_args = {'grid_coords':None,
                'shape':shape,
                'origin':geometry_origin,
                'resolution':resolution,
                'win':None,
                'geometry':geometry,
                'alg':rasterize_kwargs['alg'],
                'reduce':False}
        if merge:
            # The out buffer has already been used by the mask_in associated
            # process.
            # We have to allocate a new buffer fo rasterize => let the method
            # do it internally.
            grid_rasterize_args['output'] = None
            grid_rasterize_args['dtype'] = out.dtype
            rasterize_out = grid_rasterize(**grid_rasterize_args)
            
            # Merge : it is masked if masked by the mask_in or by the geometry
            out[:,:] |= rasterize_out[:,:]
        else:
            grid_rasterize_args['output'] = out
            _ = grid_rasterize(**grid_rasterize_args)
    
    return ret